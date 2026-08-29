"""
SPECTRA GUARD -- camera projection and pan-tilt gimbal plant.

Two things live here: the geometric map between the sky and the focal plane, and
the dynamic model of the gimbal that carries the camera.

Pinhole projection
------------------
Focal length in pixels from the horizontal field of view:

    f_px = (width/2) / tan(fov_x / 2)

A target at bearing (az, el) seen by a gimbal pointing at (pan, tilt) lands at

    u = cx + f_px * tan(az  - pan)
    v = cy - f_px * tan(el  - tilt)

The **full tan() form is retained deliberately** rather than the small-angle
approximation u ~ cx + f_px*(az-pan). During SEARCH the pointing error can be
several degrees -- comparable to the field of view itself -- and near the frame
edge tan(x) already exceeds x by

    x = 2 deg  ->  0.04%      (negligible)
    x = 20 deg ->  4.1%       (26 px at f_px = 18333... but off-frame anyway)

More importantly the tan form is what makes the *inverse* map exact, and the
EKF's measurement Jacobian is the derivative of this exact expression:

    du/d(az) = f_px * sec^2(az - pan)
    dv/d(el) = -f_px * sec^2(el - tilt)

Using a small-angle projection for rendering and a sec^2 Jacobian for the filter
would be an inconsistency the filter would interpret as model error, inflating
NIS for no physical reason. Note the minus sign on v: pixel y increases
*downward* (OpenCV / Canvas2D), while elevation increases upward.

The sign convention on u assumes azimuth increases toward image +x. Combined
with az = atan2(E, N) this makes the rendered image a view looking outward along
the boresight with East to the right when facing North -- i.e. a normal
"through the telescope" view, not a mirror image.

Gimbal plant
------------
A real coarse gimbal is a rate-commanded servo with its own closed-loop
dynamics, not a position actuator that instantly obeys. Modelled as a
first-order rate loop with hard limits:

    tau * omega_dot = -omega + K * u_cmd
    |omega_dot| <= accel_max          (torque / current limit)
    |omega|     <= slew_max           (motor back-EMF / design limit)
    angle_dot   = omega

The lag tau is what makes the control problem non-trivial: it introduces phase
lag that limits how much proportional gain the loop can carry before the
resonant peak of the sensitivity function starts amplifying platform jitter
instead of rejecting it. That trade-off is precisely what the adaptive tuner
navigates.

Three imperfections are modelled on top, because each is a real error source
that a simulation without them would hide:

  * **Jitter disturbance** is added to the *achieved* pointing, not to the
    command. A structural mode shakes the optical head regardless of what the
    rate loop is doing, so it is an output disturbance and the loop can only
    reject it to the extent its bandwidth allows.
  * **Encoder quantisation** is applied to the *reported* angle. The controller
    and the EKF read the encoder, not the truth.
  * The reported angle deliberately **excludes** the jitter. A shaft encoder
    measures shaft rotation; it does not sense the flexure of the optical bench
    downstream of it. So the jitter is unobservable to the controller and shows
    up as apparent measurement noise on the beacon centroid -- which is exactly
    how it manifests on real hardware, and why R has to be inflated for it
    (see estimation/ekf.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..config import CameraConfig


@dataclass
class GimbalTruth:
    """
    Gimbal state. ``*_cmd`` is the shaft angle the servo actually reached;
    ``*_achieved`` includes the structural jitter disturbance; ``*_reported`` is
    what the encoder tells the controller.
    """

    pan: float = 0.0
    tilt: float = 0.0
    pan_rate: float = 0.0
    tilt_rate: float = 0.0
    pan_achieved: float = 0.0
    tilt_achieved: float = 0.0
    pan_reported: float = 0.0
    tilt_reported: float = 0.0


class PinholeCamera:
    """Stateless projection helper. All angles in radians, pixels in floats."""

    def __init__(self, cfg: CameraConfig) -> None:
        self.cfg = cfg
        self.f_px = cfg.f_px
        self.cx = cfg.cx
        self.cy = cfg.cy
        self.width = cfg.width
        self.height = cfg.height
        # Beyond ~80 deg off-axis tan() blows up and the projection is
        # meaningless; clamp so a wild SEARCH excursion cannot produce inf.
        self._max_off_axis = math.radians(80.0)

    def project(self, az: float, el: float, pan: float, tilt: float) -> Tuple[float, float]:
        """
        Sky bearing -> pixel. Returns ``(u, v)``, possibly outside the frame.

        Off-frame results are returned rather than clipped: the scene renderer
        needs to know the target is out of view, and the EKF's predicted
        measurement is legitimately off-frame while coasting.
        """
        dax = _clamp(_wrap_angle(az - pan), self._max_off_axis)
        dev = _clamp(_wrap_angle(el - tilt), self._max_off_axis)
        u = self.cx + self.f_px * math.tan(dax)
        v = self.cy - self.f_px * math.tan(dev)
        return u, v

    def unproject(self, u: float, v: float, pan: float, tilt: float) -> Tuple[float, float]:
        """
        Pixel -> sky bearing. The exact inverse of ``project``.

            az = pan  + atan( (u - cx) / f_px )
            el = tilt - atan( (v - cy) / f_px )

        Used to seed the EKF from the first detection, and by any diagnostic
        that wants to express a pixel error as an angle.
        """
        az = _wrap_angle(pan + math.atan((u - self.cx) / self.f_px))
        el = _clamp(_wrap_angle(tilt - math.atan((v - self.cy) / self.f_px)), math.radians(89.5))
        return az, el

    def jacobian(self, az: float, el: float, pan: float, tilt: float) -> Tuple[float, float]:
        """
        Projection Jacobian ``(du/d_az, dv/d_el)`` at the given geometry.

            du/d(az) =  f_px * sec^2(az  - pan)
            dv/d(el) = -f_px * sec^2(el - tilt)

        Off-diagonal terms are exactly zero: with an ideal pan-tilt mount the
        two axes are orthogonal and azimuth error cannot move the image
        vertically. (A real mount has a non-orthogonality error of order
        arc-minutes; modelling it would add a small fixed cross-coupling and is
        left as future work.)
        """
        dax = _clamp(_wrap_angle(az - pan), self._max_off_axis)
        dev = _clamp(_wrap_angle(el - tilt), self._max_off_axis)
        sec2_a = 1.0 / (math.cos(dax) ** 2)
        sec2_e = 1.0 / (math.cos(dev) ** 2)
        return self.f_px * sec2_a, -self.f_px * sec2_e


    def in_frame(self, u: float, v: float, margin: float = 0.0) -> bool:
        return (-margin <= u < self.width + margin) and (-margin <= v < self.height + margin)

    def px_per_rad(self) -> float:
        """On-axis plate scale [px/rad]. Reporting helper."""
        return self.f_px


def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def _clamp(x: float, lim: float) -> float:
    return lim if x > lim else (-lim if x < -lim else x)


class Gimbal:
    """
    Pan-tilt gimbal with a first-order rate loop, slew/accel limits, an output
    jitter disturbance and encoder quantisation.

    ``step()`` integrates one control period. It is called once per frame with
    the rate command produced by the PID.
    """

    def __init__(self, cfg: CameraConfig, pan0: float = 0.0, tilt0: float = 0.0) -> None:
        self.cfg = cfg
        self.pan = float(pan0)
        self.tilt = float(tilt0)
        self.pan_rate = 0.0
        self.tilt_rate = 0.0
        self._jitter = (0.0, 0.0)

    def reset(self, pan: float = 0.0, tilt: float = 0.0) -> None:
        self.pan = float(pan)
        self.tilt = float(tilt)
        self.pan_rate = 0.0
        self.tilt_rate = 0.0
        self._jitter = (0.0, 0.0)

    def step(
        self,
        pan_rate_cmd: float,
        tilt_rate_cmd: float,
        dt: float,
        jitter_pan: float = 0.0,
        jitter_tilt: float = 0.0,
    ) -> GimbalTruth:
        """
        Advance the plant one step.

        Integration is semi-implicit: the rate is updated first, then the angle
        is advanced with the *new* rate. For a first-order lag this is more
        stable than explicit Euler at the same step size, and at dt = 16.7 ms
        against tau = 35 ms (dt/tau = 0.48) explicit Euler would already be
        visibly under-damped.
        """
        cfg = self.cfg
        self.pan_rate = self._axis_rate(self.pan_rate, pan_rate_cmd, dt)
        self.tilt_rate = self._axis_rate(self.tilt_rate, tilt_rate_cmd, dt)
        self.pan += self.pan_rate * dt
        self.tilt += self.tilt_rate * dt

        # Tilt is physically bounded; an elevation axis cannot pass through
        # zenith on a two-axis mount without a keyhole manoeuvre.
        lim = math.radians(88.0)
        if self.tilt > lim:
            self.tilt = lim
            self.tilt_rate = 0.0
        elif self.tilt < -lim:
            self.tilt = -lim
            self.tilt_rate = 0.0

        self._jitter = (jitter_pan, jitter_tilt)
        q = cfg.encoder_lsb_rad
        return GimbalTruth(
            pan=self.pan,
            tilt=self.tilt,
            pan_rate=self.pan_rate,
            tilt_rate=self.tilt_rate,
            pan_achieved=self.pan + jitter_pan,
            tilt_achieved=self.tilt + jitter_tilt,
            # Encoder sees shaft angle only -- no jitter -- quantised to 1 LSB.
            pan_reported=_quantise(self.pan, q),
            tilt_reported=_quantise(self.tilt, q),
        )

    def _axis_rate(self, omega: float, u_cmd: float, dt: float) -> float:
        """
        One axis of the rate loop:  tau*omega_dot = -omega + K*u_cmd.

        Limits are applied in physical order -- acceleration first (it is a
        torque limit, which acts on the derivative) then velocity (a design
        limit on the state). Applying them the other way round would let a
        single step jump past the accel limit whenever the velocity clamp bit.
        """
        cfg = self.cfg
        u = _clamp(u_cmd, cfg.slew_max_rad_s)
        omega_dot = (-omega + cfg.gimbal_gain * u) / cfg.gimbal_tau_s
        omega_dot = _clamp(omega_dot, cfg.accel_max_rad_s2)
        omega = omega + omega_dot * dt
        return _clamp(omega, cfg.slew_max_rad_s)

    @property
    def jitter(self) -> Tuple[float, float]:
        return self._jitter


def _quantise(x: float, lsb: float) -> float:
    if lsb <= 0.0:
        return x
    return float(np.round(x / lsb) * lsb)


__all__ = ["GimbalTruth", "PinholeCamera", "Gimbal"]
