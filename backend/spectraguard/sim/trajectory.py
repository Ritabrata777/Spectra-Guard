"""
SPECTRA GUARD -- target trajectory models.

Each model returns an analytic ``TargetState`` (bearing, angular rates, range)
as a function of time. The four kinds span the dynamic range a coarse PAT stage
has to cope with, and they exist specifically to justify design decisions
elsewhere in the stack -- a trajectory nobody's estimator struggles with proves
nothing.

  LEO_SATELLITE  Fast but perfectly smooth and predictable. This is the case a
                 constant-velocity filter handles well; it is the baseline.
  UAV_ORBIT      Slow, but with stochastic (Ornstein-Uhlenbeck) wind
                 perturbation -- unpredictable in detail, bounded in extent.
  UAV_EVASIVE    Aggressive jinking with step manoeuvres. Deliberately built to
                 defeat a constant-velocity predictor. This is the trajectory
                 that justifies the constant-ACCELERATION EKF.
  STATIC_TEST    Fixed bearing, for boresight calibration and deterministic
                 regression tests.

Coordinate convention
---------------------
Local East-North-Up (ENU) at the observer. Bearing is

    az = atan2(E, N)                 (0 = North, +ve toward East)
    el = atan2(U, sqrt(E^2 + N^2))
    range = |L|

Angular rates are obtained by central finite difference,

    d(az)/dt ~ [az(t+h) - az(t-h)] / (2h),     h = 10 ms

rather than by symbolic differentiation. This is a deliberate engineering
choice, not laziness. The exact analytic derivative of the LEO pass geometry
through the atan2 chain is long and easy to get subtly wrong, and a wrong truth
*rate* is far more insidious than a wrong truth *position*: it would corrupt the
EKF consistency assessment while leaving the picture looking correct. A central
difference is second-order accurate, O(h^2) ~ 1e-4 relative here, which is
orders of magnitude below every other error in the loop, and it cannot
disagree with the position model because it is derived from it.

The az branch cut is handled by unwrapping the difference into (-pi, pi] before
dividing, so a pass crossing due North does not produce a 2*pi/(2h) rate spike.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..config import TrajectoryConfig, TrajectoryKind

#: Earth equatorial radius [km].
EARTH_RADIUS_KM: float = 6378.137
#: Earth gravitational parameter [km^3/s^2].
MU_EARTH: float = 398600.4418


@dataclass
class TargetState:
    """Ground-truth target state at one instant. Radians, radians/s, metres."""

    t: float
    az: float
    el: float
    az_rate: float
    el_rate: float
    range_m: float

    @property
    def angular_rate(self) -> float:
        """
        Total on-sky angular rate [rad/s].

        Note the cos(el) factor: azimuth rate is a rate *about the local
        vertical*, and near zenith a large az_rate corresponds to a small
        on-sky motion. Omitting it would wildly overstate the slew demand on a
        near-zenith pass -- which is exactly the case we care about.
        """
        return math.hypot(self.az_rate * math.cos(self.el), self.el_rate)


def _wrap_pi(a: float) -> float:
    """Wrap an angle difference into (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class Trajectory:
    """
    Analytic target motion model.

    ``state(t)`` is the only entry point. Models that carry stochastic state
    (the OU gust in UAV_ORBIT, the random step manoeuvres in UAV_EVASIVE) are
    made deterministic in ``t`` by pre-generating their random sequence at
    construction and interpolating it, rather than integrating a process
    incrementally.

    Why that matters: ``state(t)`` is called at t-h and t+h for the finite
    difference, and would also be called out of order by any offline scoring or
    replay tool. A model that mutated internal state on every call would return
    different answers for the same t and make the finite-difference rate
    meaningless. Pre-generation keeps the model a pure function of time.
    """

    def __init__(
        self,
        cfg: TrajectoryConfig,
        kind: Optional[TrajectoryKind] = None,
        rng: Optional[np.random.Generator] = None,
        duration_s: float = 600.0,
    ) -> None:
        self.cfg = cfg
        self.kind: TrajectoryKind = kind if kind is not None else cfg.kind
        self._rng = rng if rng is not None else np.random.default_rng()
        self._duration = float(duration_s)

        # Finite-difference half-step. 10 ms is small against the fastest
        # modelled dynamics (1.31 Hz jink => 760 ms period) yet large enough
        # that float64 cancellation is irrelevant.
        self._h = 0.01

        self._setup()

    # ------------------------------------------------------------------ setup
    def _setup(self) -> None:
        cfg = self.cfg
        k = self.kind

        if k == "LEO_SATELLITE":
            self._r_s_km = EARTH_RADIUS_KM + cfg.leo_altitude_km
            # Mean motion n = sqrt(mu / a^3). At 550 km this is 1.0966e-3 rad/s
            # (period 95.5 min) and the satellite's inertial speed is
            # n*a = 7.59 km/s.
            self._n_orbit = math.sqrt(MU_EARTH / self._r_s_km ** 3)
            self._gamma_min = self._solve_gamma_for_elevation(math.radians(cfg.leo_max_elev_deg))
            self._build_leo_frame()

        elif k == "UAV_ORBIT":
            # OU gust sequence, pre-generated on a 50 Hz grid and interpolated.
            # See _make_ou for why the grid is fine AND the result is smoothed.
            self._ou_dt = 0.02
            n = int(self._duration / self._ou_dt) + 4
            self._ou = self._make_ou(n, self._ou_dt, cfg.uav_ou_tau_s, cfg.uav_ou_sigma_m)

        elif k == "UAV_EVASIVE":
            # Random phases make each run's jink pattern distinct while the
            # spectral content -- hence the acceleration statistics -- is fixed.
            self._phases = self._rng.uniform(0.0, 2.0 * math.pi, size=len(cfg.evasive_freqs_hz))
            self._steps = self._make_steps()

    # ------------------------------------------------------ LEO pass geometry
    def _solve_gamma_for_elevation(self, el_target: float) -> float:
        """
        Central angle gamma giving elevation ``el_target`` at closest approach.

        Elevation as a function of Earth-central angle gamma between observer and
        sub-satellite point, for orbital radius r_s and Earth radius Re:

            el(gamma) = atan2( r_s*cos(gamma) - Re,  r_s*sin(gamma) )

        This is monotonically decreasing in gamma over the visible arc, so a
        plain bisection is robust and needs no derivative -- and, importantly,
        no SciPy. 60 iterations takes gamma to ~1e-18 rad, far beyond need.
        """
        lo, hi = 1e-6, math.acos(EARTH_RADIUS_KM / self._r_s_km)  # horizon
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            el = math.atan2(
                self._r_s_km * math.cos(mid) - EARTH_RADIUS_KM, self._r_s_km * math.sin(mid)
            )
            if el > el_target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _build_leo_frame(self) -> None:
        """
        Build the orbit-plane basis for the modelled pass.

        Work in a frame with the observer at the north pole, so ENU is simply
        East=(1,0,0), North=(0,1,0), Up=(0,0,1). Then:

          * ``c`` is the unit vector to the closest-approach point, at central
            angle gamma_min from the observer, in the direction of the
            closest-approach azimuth A.
          * ``d`` is the unit velocity direction there. At closest approach the
            ground track is perpendicular to the observer-to-target great
            circle, so d is orthogonal to both c and the normal of the (o, c)
            plane.

        The satellite unit vector then sweeps the great circle

            s(theta) = cos(theta)*c + sin(theta)*d,     theta = n*(t - t_ca)

        which is exact circular-orbit motion, not an approximation.

        Getting ``d`` right is subtle and worth spelling out, because the
        plausible-looking choice is wrong. ``d`` must be perpendicular to the
        (observer, c) plane, i.e. ``d = normalise(up x c)``. Then

            cos gamma(theta) = s . up = cos(theta) cos(gamma_min)

        which is the standard great-circle pass relation with its minimum
        central angle -- hence maximum elevation -- exactly at theta = 0.

        The tempting alternative ``d = normalise(nrm x c)`` lies *inside* the
        (up, c) plane, and gives d.up = -sin(gamma_min), whence

            cos gamma(theta) = cos(theta + gamma_min)   =>   gamma = theta + gamma_min

        That is a track heading radially away from the observer's zenith: the
        elevation is highest at t=0 and falls monotonically, so the configured
        maximum elevation is never attained and there is no closest approach at
        all. The symptom is subtle -- the pass still looks smooth and the peak
        angular rate still lands in the right ballpark -- which is exactly why
        it is called out here.
        """
        g = self._gamma_min
        a = math.radians(self.cfg.leo_pass_az_deg)
        east = np.array([1.0, 0.0, 0.0])
        north = np.array([0.0, 1.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])

        c = math.cos(g) * up + math.sin(g) * (math.sin(a) * east + math.cos(a) * north)
        c /= np.linalg.norm(c)

        # Perpendicular to the (up, c) plane and tangent to the sphere at c,
        # since (up x c) . c == 0. This is the along-track direction.
        d = np.cross(up, c)
        nn = np.linalg.norm(d)
        if nn < 1e-12:
            # Exact zenith pass: (up, c) is degenerate, any tangent will do.
            d = np.cross(c, east)
            nn = np.linalg.norm(d)
        d /= nn

        self._leo_c = c
        self._leo_d = d
        self._leo_basis = (east, north, up)

    # ------------------------------------------------- stochastic components
    def _make_ou(self, n: int, dt: float, tau: float, sigma: float) -> np.ndarray:
        """
        Pre-generate a 2-axis smoothed Ornstein-Uhlenbeck sequence, shape (n, 2).

        The OU process dX = -(X/tau)dt + sigma*sqrt(2/tau)dW has the *exact*
        discrete update

            X_{k+1} = X_k * exp(-dt/tau) + sigma*sqrt(1 - exp(-2 dt/tau)) * z

        which is used rather than an Euler-Maruyama step. Exactness matters
        because it guarantees the stationary standard deviation is exactly
        ``sigma`` for any dt; an Euler step's variance is dt-dependent, so the
        gust amplitude would silently change if the grid changed.

        OU rather than white noise because wind gusts are correlated over
        seconds. A white positional perturbation would be physically absurd
        (infinite velocity) and, as with turbulence, trivially filtered out.

        *** WHY THE OU IS THEN LOW-PASSED -- A REAL BUG THIS FIXED ***

        A raw OU process is a *position* driven by white noise, so it is nowhere
        differentiable: its formal velocity is infinite, and its increments over
        a grid step dt have standard deviation

            sigma * sqrt(1 - exp(-2 dt/tau))   ~  sigma * sqrt(2 dt / tau)

        Sampling it on a grid and interpolating linearly turns that into an
        *apparent* velocity of sigma*sqrt(2/(tau*dt)), which **grows without
        bound as the grid is refined**. With the first cut of this module
        (sigma = 22 m, tau = 2.5 s, dt = 0.1 s) that was 6.1 m per 0.1 s step,
        i.e. an apparent 61 m/s gust velocity -- half again the UAV's own
        airspeed. Measured on-sky rate for UAV_ORBIT came out at 4.68 deg/s when
        the loiter geometry alone predicts ~1.5 deg/s, and the excess was pure
        interpolation artefact rather than physics.

        The fix is not a finer grid (that makes it worse) but a model with the
        right smoothness. A gust *load* is broadband, but an airframe's positional
        *response* to it is filtered by the vehicle's own mass and control loop,
        so real position histories are at least once differentiable. Two cascaded
        one-pole filters give exactly that: the output is C^1, its derivative is
        bounded by ~sigma/tau_s, and the spectrum rolls off as f^-2 above
        1/(2*pi*tau_s) instead of the OU's f^-1 tail.

        The variance lost to smoothing is restored by rescaling to the measured
        sample standard deviation, so ``uav_ou_sigma_m`` keeps meaning "RMS
        positional excursion in metres" -- the quantity a reviewer can check
        against a real flight log.
        """
        rho = math.exp(-dt / tau)
        s = sigma * math.sqrt(1.0 - rho * rho)
        z = self._rng.standard_normal((n, 2)) * s
        out = np.empty((n, 2))
        out[0] = self._rng.standard_normal(2) * sigma  # start in the stationary dist.
        for i in range(1, n):
            out[i] = rho * out[i - 1] + z[i]

        # Two cascaded one-pole low-passes -> C^1 position history.
        # tau_s = 0.8 s caps the gust-induced velocity at ~sigma/tau_s ~ 20 m/s
        # for sigma = 16 m, which is a believable gust response for a small UAV.
        tau_s = 0.8
        a = math.exp(-dt / tau_s)
        for _ in range(2):
            # scipy.signal.lfilter would be one call, but scipy is optional here
            # and n ~ 30k so the Python loop costs ~15 ms once at construction.
            acc = out[0].copy()
            for i in range(n):
                acc = a * acc + (1.0 - a) * out[i]
                out[i] = acc

        # Restore the requested RMS excursion (smoothing removed variance).
        rms = float(np.sqrt(np.mean(out ** 2)))
        if rms > 1e-9:
            out *= sigma / rms
        return out

    def _make_steps(self) -> np.ndarray:
        """
        Pre-generate step-manoeuvre times and lateral offsets, shape (k, 3).

        Columns are ``(t_start, dx, dy)``. Inter-arrival times are exponential
        (a Poisson process in time), so manoeuvres are genuinely unpredictable
        rather than periodic -- a periodic manoeuvre would eventually be learned
        by the filter's acceleration state, which would be cheating in our
        favour.
        """
        cfg = self.cfg
        times = []
        t = float(self._rng.exponential(cfg.evasive_step_interval_s))
        while t < self._duration:
            times.append(t)
            t += float(self._rng.exponential(cfg.evasive_step_interval_s))
        if not times:
            return np.zeros((0, 3))
        arr = np.zeros((len(times), 3))
        arr[:, 0] = times
        ang = self._rng.uniform(0.0, 2.0 * math.pi, size=len(times))
        arr[:, 1] = cfg.evasive_step_m * np.cos(ang)
        arr[:, 2] = cfg.evasive_step_m * np.sin(ang)
        return arr

    # ------------------------------------------------------- position models
    def _enu(self, t: float) -> Tuple[float, float, float]:
        """Target position in local ENU [m] at time t. Pure function of t."""
        k = self.kind
        cfg = self.cfg

        if k == "STATIC_TEST":
            r = cfg.static_range_km * 1000.0
            az = math.radians(cfg.static_az_deg)
            el = math.radians(cfg.static_el_deg)
            return (
                r * math.cos(el) * math.sin(az),
                r * math.cos(el) * math.cos(az),
                r * math.sin(el),
            )

        if k == "LEO_SATELLITE":
            theta = self._n_orbit * (t - cfg.leo_tca_s)
            s = math.cos(theta) * self._leo_c + math.sin(theta) * self._leo_d
            # Kilometres -> metres, minus the observer's own radius vector.
            p = s * self._r_s_km - np.array([0.0, 0.0, EARTH_RADIUS_KM])
            return float(p[0] * 1000.0), float(p[1] * 1000.0), float(p[2] * 1000.0)

        if k == "UAV_ORBIT":
            omega = cfg.uav_speed_m_s / max(cfg.uav_loiter_radius_m, 1.0)
            cx, cy = 0.0, cfg.uav_range_m
            e = cx + cfg.uav_loiter_radius_m * math.cos(omega * t)
            n = cy + cfg.uav_loiter_radius_m * math.sin(omega * t)
            gx, gy = self._ou_at(t)
            return e + gx, n + gy, cfg.uav_alt_m

        if k == "UAV_EVASIVE":
            # Base motion: a straight crossing pass, so the jink is not confused
            # with the loiter circle's own curvature.
            e = -600.0 + cfg.uav_speed_m_s * 0.55 * t
            n = cfg.uav_range_m
            # Sum-of-sines jink applied laterally (cross-track = North here).
            jx = 0.0
            jy = 0.0
            for f, amp, ph in zip(cfg.evasive_freqs_hz, cfg.evasive_amps_m, self._phases):
                w = 2.0 * math.pi * f
                jx += amp * math.sin(w * t + ph)
                jy += amp * 0.55 * math.cos(w * t + ph * 1.7)
            sx, sy = self._steps_at(t)
            return e + jx + sx, n + jy + sy, cfg.uav_alt_m

        raise ValueError(f"unknown trajectory kind {k!r}")

    def _ou_at(self, t: float) -> Tuple[float, float]:
        """Linearly interpolate the pre-generated OU gust at time t."""
        if t <= 0.0:
            return float(self._ou[0, 0]), float(self._ou[0, 1])
        x = t / self._ou_dt
        i = int(x)
        if i >= self._ou.shape[0] - 1:
            return float(self._ou[-1, 0]), float(self._ou[-1, 1])
        f = x - i
        a = self._ou[i]
        b = self._ou[i + 1]
        return float(a[0] + f * (b[0] - a[0])), float(a[1] + f * (b[1] - a[1]))

    def _steps_at(self, t: float) -> Tuple[float, float]:
        """
        Accumulated step-manoeuvre displacement at time t.

        Each step is smoothed by a raised-cosine over ``ramp`` seconds instead
        of being a true discontinuity. A position step would imply infinite
        velocity and would show up as a single-frame teleport that no filter
        could or should follow.

        The ramp duration is a real design parameter, not cosmetic. A raised
        cosine of total displacement d over duration T has peak velocity
        (pi/2)*d/T, so a 60 m step in 0.35 s peaks at 269 m/s -- supersonic, and
        it alone contributed 5.1 deg/s to the measured on-sky rate at 3 km. At
        0.5 s the peak is 188 m/s: still a violent manoeuvre for a UAV, but it
        no longer dominates the rate budget, and the *acceleration* content --
        which is what the constant-acceleration EKF is being tested against --
        is barely affected because the manoeuvre's bandwidth is unchanged.
        """
        if self._steps.shape[0] == 0:
            return 0.0, 0.0
        ramp = 0.5
        ts = self._steps[:, 0]
        # Vectorised raised-cosine ramp: 0 before, 1 after, smooth in between.
        u = np.clip((t - ts) / ramp, 0.0, 1.0)
        w = 0.5 - 0.5 * np.cos(math.pi * u)
        return float(np.dot(w, self._steps[:, 1])), float(np.dot(w, self._steps[:, 2]))

    # ------------------------------------------------------------------ state
    def _bearing(self, t: float) -> Tuple[float, float, float]:
        e, n, u = self._enu(t)
        horiz = math.hypot(e, n)
        az = math.atan2(e, n)
        el = math.atan2(u, horiz)
        rng = math.sqrt(e * e + n * n + u * u)
        return az, el, rng

    def state(self, t: float) -> TargetState:
        """Ground-truth target state at time ``t``. Pure function of t."""
        az, el, rng = self._bearing(t)
        h = self._h
        az_m, el_m, _ = self._bearing(t - h)
        az_p, el_p, _ = self._bearing(t + h)
        # Unwrap the azimuth difference so a pass crossing due North does not
        # produce a spurious 2*pi/(2h) rate spike.
        az_rate = _wrap_pi(az_p - az_m) / (2.0 * h)
        el_rate = (el_p - el_m) / (2.0 * h)
        return TargetState(
            t=t, az=az, el=el, az_rate=az_rate, el_rate=el_rate, range_m=rng
        )

    def peak_angular_rate(self, t0: float = 0.0, t1: float = 30.0, n: int = 600) -> float:
        """Sample the on-sky rate over a window -- a reporting/validation helper."""
        return max(self.state(t).angular_rate for t in np.linspace(t0, t1, n))


__all__ = ["EARTH_RADIUS_KM", "MU_EARTH", "TargetState", "Trajectory"]
