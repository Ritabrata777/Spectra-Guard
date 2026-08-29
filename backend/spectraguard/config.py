"""
SPECTRA GUARD -- centralised, immutable configuration.

Every tunable in the simulator, estimator and controller lives here as a frozen
dataclass. Nothing else in the package is allowed to hard-code a magic number
that an ISRO reviewer might want to change; if you find one, it belongs in this
file. Frozen (immutable) dataclasses are used deliberately: a configuration
object is captured in the run log and must describe the run that actually
happened, so it must not be mutated half-way through by a slider callback.

Operator-adjustable *disturbances* are the one exception -- those live in
``Disturbance`` and are patched live by the UI, which is why they are a separate
mutable-by-replacement object rather than part of ``AppConfig``.

Unit convention (matches docs/PROTOCOL.md)
-----------------------------------------
Internally, Python works in SI + radians everywhere. Degrees appear ONLY at the
wire boundary in ``schema.py``. This single rule removes an entire class of
bug: if you see ``math.degrees`` outside ``schema.py`` or a plotting helper,
it is probably wrong.

  angles          radians
  angular rates   radians / second
  Cn2             m^(-2/3)
  wavelength      metres
  path length     metres
  pixel coords    float pixels, origin top-left, +y DOWN (OpenCV / Canvas2D)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Literal, Tuple


__all__ = [
    "LOCK_TOLERANCE_PX",
    "COAST_TIMEOUT_FRAMES",
    "CHI2_GATE_2DOF_99",
    "TrajectoryKind",
    "TunerMode",
    "DetectorBackend",
    "PatStage",
    "CameraConfig",
    "TurbulenceConfig",
    "JitterConfig",
    "NoiseConfig",
    "SceneConfig",
    "EkfConfig",
    "PidConfig",
    "TunerConfig",
    "DetectorConfig",
    "PatConfig",
    "Disturbance",
    "TrajectoryConfig",
    "AppConfig",
    "DEFAULT_CONFIG",
]


# --------------------------------------------------------------------------- #
#  cross-cutting constants
# --------------------------------------------------------------------------- #

#: Pixel miss distance inside which the coarse stage declares lock. Mirrored in
#: frontend/lib/types.ts. At the default plate scale this is ~654 urad, which is
#: a realistic capture range for a fine steering mirror -- so this constant *is*
#: the engineering requirement the whole system exists to satisfy.
LOCK_TOLERANCE_PX = 12.0

#: Consecutive blind frames before COAST gives up and returns to SEARCH.
#: 90 frames = 1.5 s at 60 Hz. Past this the posterior covariance has grown
#: wider than the a-priori search cone, so re-scanning is genuinely cheaper than
#: continuing to extrapolate.
COAST_TIMEOUT_FRAMES = 90

#: Chi-square 99th percentile, 2 degrees of freedom. The NIS gate: a normalised
#: innovation above this is rejected as an outlier rather than fused. Two DOF
#: because the measurement is a 2-vector (u, v).
CHI2_GATE_2DOF_99 = 9.2103


TrajectoryKind = Literal["LEO_SATELLITE", "UAV_ORBIT", "UAV_EVASIVE", "STATIC_TEST"]
TunerMode = Literal["FIXED", "SAC"]
DetectorBackend = Literal["YOLOV11", "CLASSICAL"]
PatStage = Literal["IDLE", "SEARCH", "ACQUIRE", "TRACK", "COAST", "FAULT"]


# --------------------------------------------------------------------------- #
#  camera + gimbal plant
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CameraConfig:
    """
    Coarse-acquisition camera intrinsics and the gimbal plant it rides on.

    The pinhole focal length in pixels follows from the horizontal field of
    view:

        f_px = (width / 2) / tan(fov_x / 2)

    With the defaults below (640 px, 2 deg) this is ~18333 px/rad, i.e. one
    pixel subtends ~54.5 microradians. That sets the scale for everything else:
    LOCK_TOLERANCE_PX = 12 px is ~654 urad (0.0375 deg), which is a realistic
    hand-off tolerance from a coarse stage to a fine steering mirror.

    *** WHERE TURBULENCE ACTUALLY BITES AT THIS PLATE SCALE ***

    Worth stating plainly, because it is the single most common thing to get
    wrong when reading this simulation. Angle-of-arrival jitter for the default
    atmosphere (Cn2 = 1e-15 over 5 km, D = 0.2 m) is sigma_AoA = 5.0 urad, which
    at 54.5 urad/px is **0.09 px** -- two orders of magnitude below the lock
    tolerance. Turbulent tilt is *not* what limits a coarse tracker, and any
    simulation that makes it look otherwise has either an unrealistically long
    focal length or a bug.

    What turbulence actually does to this stage, in order of importance:

      1. **Scintillation fades.** sigma_I^2 = 0.14 at default, rising past 10 at
         Cn2 = 1e-13. A deep fade drops the beacon below the detection floor and
         forces COAST. This is the dominant turbulence coupling.
      2. **PSF broadening.** D/r0 = 1.7 at default, ~27 at Cn2 = 1e-13. Peak SNR
         falls as 1/sigma^2 at conserved flux, and centroid precision degrades
         roughly as sigma/SNR.
      3. **Tilt**, last. It only becomes comparable to the platform jitter
         (150 urad default = 2.75 px) at Cn2 ~ 1e-12, since sigma_AoA ~ sqrt(Cn2)
         while the jitter is set independently by the operator.

    All three are modelled. The FOV is a genuine trade: wide enough that SEARCH
    over a 3 deg a-priori cone terminates in seconds, narrow enough that
    12 px of lock tolerance is a meaningful angular requirement.
    """

    width: int = 640
    height: int = 480

    #: Horizontal field of view, degrees. See the class docstring for the trade.
    fov_deg: float = 2.0

    #: Receive aperture diameter, metres. Sets aperture averaging of
    #: scintillation, the D/r0 ratio that broadens the PSF, and the
    #: angle-of-arrival variance.
    aperture_m: float = 0.2

    #: Communication wavelength, metres. 1550 nm: eye-safe at far higher power
    #: than 1064 nm, low atmospheric absorption, and the band every practical
    #: FSOC terminal actually uses because the telecom industry already built
    #: the components.
    wavelength_m: float = 1.55e-06

    #: Gimbal rate-loop time constant, seconds. A real coarse gimbal is a
    #: first-order lag behind its rate command, not an ideal integrator (~4.5 Hz
    #: here). Modelling the lag is what makes the derivative term earn its keep;
    #: a PID tuned against an ideal plant falls apart on a real one.
    gimbal_tau_s: float = 0.035

    #: Rate-loop DC gain. 1.0 = perfectly calibrated. Detune it to see the
    #: integral term work for its living.
    gimbal_gain: float = 1.0

    #: Slew rate limit, rad/s (~28.6 deg/s).
    slew_max_rad_s: float = 0.5

    #: Angular acceleration limit, rad/s^2 (~115 deg/s^2). Finite torque: the
    #: reason a step command cannot be answered instantly, and therefore the
    #: reason acquisition takes a measurable time at all.
    accel_max_rad_s2: float = 2.0

    #: Encoder quantisation, radians (~10 urad, a 19-bit encoder over +/-180 deg).
    #: 0.18 px here -- small, but it puts a hard floor under pointing knowledge
    #: that no amount of filtering can dig below.
    encoder_lsb_rad: float = 4.8e-05

    @property
    def fov_rad(self) -> float:
        return math.radians(self.fov_deg)

    @property
    def f_px(self) -> float:
        """Focal length in pixels, from the horizontal FOV."""
        return (self.width / 2.0) / math.tan(self.fov_rad / 2.0)

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    @property
    def wavenumber(self) -> float:
        """k = 2*pi/lambda  [rad/m]."""
        return 2.0 * math.pi / self.wavelength_m


# --------------------------------------------------------------------------- #
#  atmosphere
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TurbulenceConfig:
    """
    Atmospheric turbulence model parameters (see physics/turbulence.py).

    ``screen_n`` and ``screen_dx_m`` set the spatial bandwidth of the phase
    screen. The grid resolves spatial frequencies from
    dkappa = 2*pi/(N*dx) up to the Nyquist pi/dx. With N=256 and dx=5 mm the
    screen is 1.28 m across and the aperture (0.2 m) spans 40 samples --
    enough to fit a tilt plane meaningfully without wasting FFT time.
    """

    #: Slant path length, metres. 5 km is a representative mobile terminal
    #: geometry (vehicle-to-mast or vehicle-to-vehicle) and is where the
    #: turbulence numbers stop being negligible.
    path_length_m: float = 5000.0

    #: von Karman outer scale L0, metres. The largest eddy that still obeys
    #: Kolmogorov statistics; above it the cascade breaks down. 50 m is typical
    #: for a near-ground horizontal path. A finite outer scale matters because
    #: pure Kolmogorov has infinite power at zero frequency, which shows up as
    #: enormous unphysical tilt -- tilt a gimbal would simply track out anyway.
    outer_scale_m: float = 50.0

    #: Inner scale l0, metres. Where viscous dissipation kills the cascade.
    inner_scale_m: float = 0.005

    #: Phase screen grid size and sample spacing. See the class docstring.
    screen_n: int = 256
    screen_dx_m: float = 0.005

    #: Number of independent screens pre-generated and cycled. Regenerating a
    #: 256x256 FFT screen every frame would cost more than the entire rest of
    #: the pipeline; four screens plus frozen-flow translation gives temporal
    #: variety at negligible cost.
    n_screens: int = 4

    #: Lane/Johansson subharmonic levels restored below the FFT fundamental.
    #: Without these the structure function comes out 30-40% low. See
    #: physics/turbulence.py.
    subharmonic_levels: int = 3

    #: Transverse wind, m/s, and its direction in the screen plane. Under the
    #: Taylor frozen-flow hypothesis this is what converts the spatial statistics
    #: into temporal ones, and therefore what sets how long a fade lasts.
    wind_speed_m_s: float = 5.0
    wind_dir_rad: float = 0.35

    #: Reference Cn2 used for one-off diagnostics and the report's tables.
    cn2_ref: float = 1e-15


# --------------------------------------------------------------------------- #
#  platform vibration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class JitterConfig:
    """Platform micro-vibration model (see physics/jitter.py)."""

    #: Resonator quality factor. A vehicle or mast does not vibrate as white
    #: noise: it rings at structural modes. Q = 8 gives a recognisable narrowband
    #: peak that the adaptive tuner can actually find in the PSD, which is the
    #: whole point -- a flat spectrum would leave nothing to adapt to.
    q: float = 8.0

    #: Samples retained for the running Welch PSD estimate. 256 frames = 4.3 s
    #: at 60 Hz, giving ~0.23 Hz resolution: enough to localise the peak the
    #: tuner schedules against without lagging a genuine change in conditions.
    psd_history: int = 256

    #: Samples used for the one-off amplitude calibration at construction, so the
    #: resonator's output RMS matches the requested jitter_urad exactly rather
    #: than approximately. Analytic normalisation of a discrete biquad is
    #: error-prone; measuring it once at startup costs 8192 scalar steps.
    calibration_samples: int = 8192


# --------------------------------------------------------------------------- #
#  focal-plane sensor
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class NoiseConfig:
    """Focal-plane sensor noise (see physics/noise.py)."""

    #: Conversion gain, electrons per DN. Sets the shot-noise scale: a signal of
    #: S DN carries sqrt(S * gain) electrons of Poisson noise, i.e.
    #: sqrt(S / gain) DN. Shot noise is signal-dependent, which is why a bright
    #: beacon centroids better than a faint one even at identical read noise.
    gain_e_per_dn: float = 4.0

    #: Dark current pedestal, DN, at the modelled integration time.
    dark_current_dn: float = 2.0

    #: Read noise, DN RMS. The floor that survives however long you integrate.
    read_noise_dn: float = 1.2

    #: Draw true Poisson samples instead of the Gaussian approximation. Off by
    #: default: at these signal levels (tens to hundreds of DN, so hundreds to
    #: thousands of electrons) the Gaussian approximation is excellent and about
    #: 8x faster, which matters when the budget is 16 ms per frame.
    exact_poisson: bool = False


# --------------------------------------------------------------------------- #
#  synthetic scene
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SceneConfig:
    """Synthetic focal-plane image generation (see sim/scene.py)."""

    #: Beacon peak, DN, before scintillation and PSF broadening. 260 against a
    #: sky of 26 +/- 6 DN is a comfortable but not trivial SNR -- bright enough
    #: that the classical detector works, dim enough that a deep fade genuinely
    #: loses it.
    beacon_peak_dn: float = 260.0

    #: Diffraction-limited PSF sigma, pixels, and the cap after turbulent
    #: broadening. The cap exists because the flux-conserving peak scales as
    #: 1/sigma^2, and an unbounded sigma would silently produce a beacon of
    #: zero amplitude spread over the whole frame.
    beacon_sigma_px: float = 1.9
    beacon_sigma_max_px: float = 7.0

    #: Beacon amplitude modulation. The beacon is deliberately blinked so the
    #: receiver can distinguish it from a star, which is exactly how real
    #: acquisition beacons are discriminated. 7 Hz is chosen to be well below
    #: the 60 Hz frame rate (so it is resolved, not aliased) and not a harmonic
    #: of the 12 Hz default platform jitter (so the two cannot be confused).
    beacon_mod_hz: float = 7.0
    beacon_mod_depth: float = 0.18

    #: Decoys: bright point sources placed near the beacon, one of them
    #: *brighter* than it (1.15x). This is the single most important adversarial
    #: element in the scene. A brightest-pixel or largest-blob tracker -- which
    #: is what a basic OpenCV implementation reduces to -- locks onto the decoy
    #: and never recovers. Ours has to use shape, area and motion prior, and the
    #: decoys are how we prove it.
    n_decoys: int = 3
    decoy_peak_frac: Tuple[float, ...] = (1.15, 0.85, 0.7)

    #: Star field: clutter with a realistic brightness distribution.
    n_stars: int = 260
    star_peak_dn: Tuple[float, float] = (12.0, 55.0)

    #: Sky background (mean, sigma) in DN. The spatial sigma, not the temporal
    #: noise -- real sky is not uniform, and a detector that assumes it is will
    #: threshold badly at the edges of the frame.
    sky_dn: Tuple[float, float] = (26.0, 6.0)

    #: Cos^4 vignetting depth, 0..1. Present because it makes a global threshold
    #: fail near the corners, which is the honest reason to use a local one.
    vignette_strength: float = 0.55

    #: Cloud alpha above which ground truth declares the beacon LOST. This is
    #: the reference the COAST logic is scored against -- without a truth
    #: definition of "should have been invisible", lock retention is unfalsifiable.
    occlusion_lost_alpha: float = 0.62

    #: Cloud layer: drift speed in px/s, blob size as a fraction of the frame,
    #: edge softness, and the coarse grid the field is generated on before being
    #: resized up. Generating cloud at 60x80 and interpolating is ~60x cheaper
    #: than generating it at full resolution and visually indistinguishable.
    cloud_speed_px_s: float = 140.0
    cloud_blob_sigma_frac: float = 0.18
    cloud_edge_softness: float = 0.22
    cloud_grid: Tuple[int, int] = (60, 80)


# --------------------------------------------------------------------------- #
#  estimator
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class EkfConfig:
    """Nearly-constant-acceleration EKF in inertial bearing space."""

    #: Jerk process-noise intensity, rad/s^3 per sqrt(Hz). The single knob that
    #: says "how surprising is the target allowed to be". Too small and the
    #: filter refuses to believe a manoeuvre and lags it; too large and it
    #: chases centroid noise and its covariance never shrinks enough to be
    #: useful for coasting.
    sigma_jerk: float = 0.9

    #: Measurement noise, pixels (1 sigma), for a clean high-SNR detection.
    sigma_pix_base: float = 0.9

    #: Measurement noise inflation as detector score falls:
    #: sigma = sigma_pix_base * (1 + score_inflation * (1 - score)).
    #: This is how detector confidence enters the filter *quantitatively*
    #: instead of as a threshold. A marginal detection is still information; it
    #: just deserves less weight, and R is exactly where that belongs.
    score_inflation: float = 6.0

    #: Initial 1-sigma uncertainty at track birth: rad, rad/s, rad/s^2.
    #: Deliberately loose -- an over-confident initial covariance makes the gate
    #: reject the very measurements needed to correct it, and the track dies
    #: before it starts.
    init_sigma_pos_rad: float = 0.003
    init_sigma_vel_rad_s: float = 0.008
    init_sigma_acc_rad_s2: float = 0.02

    #: Frames of NIS history kept for the consistency statistic. A correctly
    #: tuned filter has mean NIS = 2.0 (the DOF); the running mean over 60
    #: frames is what the UI shows, and it is a real self-diagnostic rather than
    #: a decorative number.
    nis_window: int = 60

    #: Re-acquisition gate while coasting: accept a detection within
    #: max(reacquire_sigma * sigma_pred, reacquire_min_px) of the prediction.
    #: The floor matters because a confident filter's 3-sigma ellipse can shrink
    #: below the true uncertainty during a long blind coast, and then nothing
    #: would ever be allowed back in.
    reacquire_sigma: float = 3.0
    reacquire_min_px: float = 18.0


# --------------------------------------------------------------------------- #
#  controller
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PidConfig:
    """Gimbal rate-command PID (see control/pid.py)."""

    kp: float = 0.42
    ki: float = 0.06
    kd: float = 0.11

    #: Derivative low-pass cutoff, Hz. Differentiating a noisy centroid without
    #: this turns a "PID" into a noise amplifier -- the most common way a
    #: textbook controller fails on real sensor data.
    d_cutoff_hz: float = 8.0

    #: Rate-command clamp, rad/s. Matches the plant's slew limit; commanding
    #: past it just desynchronises the integrator from reality.
    out_limit_rad_s: float = 0.5

    #: Integrator clamp (anti-windup), in scaled error units. During SEARCH the
    #: error is huge and persistent; an unclamped integrator saturates and then
    #: overshoots catastrophically at the exact moment of acquisition.
    integ_limit: float = 0.05

    #: Error normalisation, px per unit controller error. Keeps the gains
    #: dimensionless and O(1), so the tuner's log-space search and the UI's
    #: gain tiles are both interpretable.
    error_scale: float = 200.0


@dataclass(frozen=True)
class TunerConfig:
    """Adaptive gain tuner (see control/sac_tuner.py)."""

    mode: TunerMode = "SAC"

    #: Gains are bounded to a multiplicative band around the PidConfig
    #: baseline, in log space. Hard bounds are non-negotiable: an RL agent given
    #: unbounded authority over a control loop will eventually find an unstable
    #: corner, and "the neural network destabilised the gimbal" is not a
    #: recoverable demo failure.
    gain_min_frac: float = 0.25
    gain_max_frac: float = 4.0

    #: Maximum change in log-gain per update, and how often updates happen.
    #: Rate-limiting the *action* rather than only bounding it means the loop
    #: never sees a step change in its own dynamics.
    max_log_step: float = 0.035
    interval_frames: int = 6

    #: Reward weights: track error, control effort, overshoot, gain thrash.
    #: Overshoot is penalised separately from error because a controller that
    #: minimises RMS error alone will happily ring, and ringing is precisely
    #: what breaks a hand-off to a fine stage.
    w_error: float = 1.0
    w_effort: float = 0.05
    w_overshoot: float = 0.35
    w_gain_change: float = 0.02

    #: SAC network and optimiser hyperparameters. Used only when torch is
    #: importable; otherwise the CEM tuner runs on numpy and says so.
    hidden: int = 128
    lr: float = 0.0003
    gamma: float = 0.97
    tau_polyak: float = 0.01
    batch_size: int = 64
    replay_capacity: int = 20000

    #: Frames of fixed-gain operation before the tuner is allowed to act. The
    #: baseline PID must be seen to work on its own first -- both because an
    #: untrained policy is worse than the hand-tuned gains, and because a jury
    #: needs the before/after comparison to believe the after.
    warmup_frames: int = 600

    target_entropy_scale: float = 1.0

    #: Nominal closed-loop bandwidth, Hz, used to seed the gain schedule and to
    #: judge which jitter frequencies are inside the loop's authority.
    loop_bandwidth_hz: float = 15.0


# --------------------------------------------------------------------------- #
#  detector
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DetectorConfig:
    """Beacon detector configuration (see vision/detector.py)."""

    #: Preferred backend. YOLOV11 is attempted first; if ultralytics or the
    #: weights are missing, the classical detector takes over and the telemetry
    #: reports CLASSICAL. The fallback is never silent.
    prefer: DetectorBackend = "YOLOV11"
    yolo_weights: str = "weights/beacon_yolov11n.pt"
    yolo_conf: float = 0.25

    #: White top-hat kernel size, px. Removes anything larger than the kernel,
    #: which is exactly the sky gradient, vignetting and cloud -- while leaving
    #: point sources untouched. This one morphological step is why a local
    #: threshold is unnecessary.
    tophat_ksize: int = 11

    #: Threshold as the stricter of a high percentile and mean + k*sigma, with a
    #: hard DN floor. Belt and braces: the percentile adapts to scene content,
    #: the sigma rule adapts to noise, and the floor stops a blank frame from
    #: thresholding at essentially zero and returning ten thousand blobs.
    thresh_percentile: float = 99.6
    thresh_k_sigma: float = 4.5
    thresh_floor_dn: float = 8.0

    #: Blob area gate, px. Below min is a hot pixel or read noise; above max is
    #: a cloud edge or lens flare, not a beacon.
    min_area_px: int = 2
    max_area_px: int = 900

    #: Expected beacon area, px, used to score candidates by plausibility
    #: rather than merely accepting them.
    expect_area_px: float = 28.0

    #: Half-window for sub-pixel centroid refinement, px.
    centroid_window: int = 5

    #: Candidate scoring weights: peak brightness, roundness, area agreement,
    #: and agreement with the EKF's predicted position.
    #:
    #: Note that w_prior (2.6) dominates the appearance terms combined. That is
    #: the deliberate architectural claim of this project: once a track exists,
    #: *where the beacon should be* is stronger evidence than *what it looks
    #: like*. It is what defeats the brighter-than-the-beacon decoy, and it is
    #: why the estimator is upstream of the final detection decision rather
    #: than downstream of it.
    w_peak: float = 1.0
    w_shape: float = 0.8
    w_area: float = 0.6
    w_prior: float = 2.6

    #: Score floor below which a candidate is reported as no detection at all.
    #: Reporting nothing and letting the EKF coast is strictly better than
    #: feeding the filter a guess, because a wrong measurement inside the gate
    #: corrupts the track while a missing one merely widens the covariance.
    min_score: float = 0.16


# --------------------------------------------------------------------------- #
#  PAT state machine
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PatConfig:
    """Pointing-Acquisition-Tracking state machine (see engine.py)."""

    target_hz: float = 60.0

    #: A-priori uncertainty cone to be searched, radians (~3 deg). This is the
    #: residual pointing error after a GPS/IMU handover, and it is why a coarse
    #: stage exists at all: a fine-tracking quad-cell with tens of microradians
    #: of capture range cannot see a 3 deg error.
    search_cone_rad: float = 0.0524

    #: Archimedean spiral radial growth per turn, radians (~1.1 deg). Set to
    #: slightly less than the FOV so consecutive turns overlap -- a spiral with
    #: pitch equal to the FOV leaves unsearched gaps at the corners.
    search_pitch_rad: float = 0.0192

    #: Spiral tangential speed, rad/s (~8 deg/s). Bounded by dwell time: sweep
    #: faster than the beacon can be integrated and detected and the scan misses
    #: it even while pointing straight at it.
    search_speed_rad_s: float = 0.14

    #: Consecutive in-tolerance frames required for ACQUIRE -> TRACK. One lucky
    #: frame is not a lock.
    lock_frames: int = 8

    #: Consecutive missed detections before TRACK -> COAST. Three, not one: a
    #: single dropped frame is normal scintillation, and thrashing the state
    #: machine on it would make the UI unreadable and the metrics meaningless.
    miss_frames_to_coast: int = 3

    #: Consecutive gated detections required to leave COAST.
    reacquire_frames: int = 3

    coast_timeout_frames: int = COAST_TIMEOUT_FRAMES

    #: Frames held in FAULT before restarting SEARCH, so the failure is visible
    #: to an operator rather than flickering past.
    fault_hold_frames: int = 45

    #: Target-lock confidence is a weighted blend of three *independent*
    #: evidence sources: how much the detector likes the measurement, how
    #: statistically consistent the filter is (NIS), and how tight the posterior
    #: covariance is. Blending matters -- detector score alone is confidently
    #: wrong when locked onto a decoy, and covariance alone is confidently tight
    #: when coasting on a stale track. Disagreement between them is the signal.
    conf_w_detector: float = 0.45
    conf_w_nis: float = 0.25
    conf_w_cov: float = 0.30

    #: Covariance scale, px, at which the covariance term is fully discounted.
    conf_cov_scale_px: float = 45.0

    #: Per-frame confidence decay while coasting (0.982^90 = 0.20 over a full
    #: timeout). Confidence must visibly bleed away during a blind coast; a
    #: system that claims high confidence with no measurements is lying, and
    #: the operator's decision to intervene depends on being told the truth.
    conf_coast_decay: float = 0.982


# --------------------------------------------------------------------------- #
#  operator disturbances
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Disturbance:
    """
    Operator-injected physical disturbances. Mirrors ``Disturbance`` in
    frontend/lib/types.ts, including the default values, which the UI sliders
    initialise from.

    Frozen, and patched by ``replace()``: the engine swaps in a whole new
    object between ticks, so a tick never observes a half-applied patch.
    """

    #: Refractive index structure constant, m^(-2/3). 1e-17 is good seeing at
    #: night over water; 1e-15 a daytime horizontal average; 1e-13 strong
    #: near-ground turbulence over hot terrain.
    cn2: float = 1e-15

    #: Dominant platform vibration frequency, Hz.
    jitter_hz: float = 12.0

    #: Platform vibration amplitude, microradians RMS. 150 urad is 2.75 px at
    #: the default plate scale: clearly visible on the trajectory plot, well
    #: inside the 12 px lock tolerance, and realistic for a vehicle-mounted
    #: terminal with modest isolation. This -- not turbulence -- is the dominant
    #: pointing disturbance, which is the point the class docstring of
    #: CameraConfig labours.
    jitter_urad: float = 150.0

    #: Additive read/thermal noise, DN RMS, on top of the modelled sensor noise.
    awgn_sigma: float = 4.0

    #: Cloud obscuration opacity, 0..1.
    occlusion: float = 0.0

    #: Wire (camelCase) -> attribute (snake_case) mapping. The one place in the
    #: backend where the two naming conventions are allowed to meet, so that the
    #: translation exists exactly once and is inspectable.
    WIRE_MAP = {
        "cn2": "cn2",
        "jitterHz": "jitter_hz",
        "jitterUrad": "jitter_urad",
        "awgnSigma": "awgn_sigma",
        "occlusion": "occlusion",
    }

    def patched(self, patch: dict) -> "Disturbance":
        """
        Apply a partial camelCase patch, ignoring unknown keys.

        Tolerating unknown keys is deliberate: a newer frontend must not be
        able to kill the engine by sending a field this build has never heard
        of. See docs/PROTOCOL.md "Commands".

        Values are clamped here rather than trusted, because this is a network
        boundary. A Cn2 of 1e300 or a NaN would propagate into r0, then into
        every derived quantity, and the failure would surface far from its
        cause -- as a frozen viewport rather than a rejected command.
        """
        kwargs: dict = {}
        for wire_key, value in (patch or {}).items():
            attr = self.WIRE_MAP.get(wire_key)
            if attr is None:
                continue
            try:
                kwargs[attr] = float(value)
            except (TypeError, ValueError):
                continue

        if "cn2" in kwargs:
            kwargs["cn2"] = min(max(kwargs["cn2"], 1e-18), 1e-12)
        if "jitter_hz" in kwargs:
            kwargs["jitter_hz"] = min(max(kwargs["jitter_hz"], 0.5), 120.0)
        if "jitter_urad" in kwargs:
            kwargs["jitter_urad"] = min(max(kwargs["jitter_urad"], 0.0), 2000.0)
        if "awgn_sigma" in kwargs:
            kwargs["awgn_sigma"] = min(max(kwargs["awgn_sigma"], 0.0), 80.0)
        if "occlusion" in kwargs:
            kwargs["occlusion"] = min(max(kwargs["occlusion"], 0.0), 1.0)

        return replace(self, **kwargs)


# --------------------------------------------------------------------------- #
#  target motion
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TrajectoryConfig:
    """Target motion model parameters (see sim/trajectory.py)."""

    kind: TrajectoryKind = "LEO_SATELLITE"

    #: LEO pass geometry. A 550 km circular orbit is the Starlink/OneWeb shell
    #: and the realistic case for an optical ground terminal. Peak angular rate
    #: at zenith is ~0.9 deg/s -- a genuine test of a coarse gimbal, and the
    #: reason velocity feed-forward from the EKF is not optional.
    leo_altitude_km: float = 550.0
    leo_max_elev_deg: float = 78.0
    leo_tca_s: float = 8.0
    leo_pass_az_deg: float = 35.0

    #: UAV loiter: a slant range of 3 km at 400 m altitude, orbiting at 30 m/s.
    uav_range_m: float = 3000.0
    uav_alt_m: float = 400.0
    uav_loiter_radius_m: float = 400.0
    uav_speed_m_s: float = 30.0

    #: Ornstein-Uhlenbeck wander superimposed on the ideal orbit: correlation
    #: time and position sigma. Real platforms do not fly analytic curves, and a
    #: filter validated only against one is validated against nothing. An OU
    #: process is the right model because it is mean-reverting -- the UAV strays
    #: from its commanded track and is pulled back, rather than random-walking
    #: away forever.
    uav_ou_tau_s: float = 2.5
    uav_ou_sigma_m: float = 16.0

    #: Evasive manoeuvre: a sum of four incommensurate sinusoids plus periodic
    #: step changes. The frequencies are deliberately not harmonics of each
    #: other, so the motion never becomes predictable to a filter that has
    #: latched onto one period, and the steps inject the acceleration
    #: discontinuities that expose an over-confident process model.
    evasive_freqs_hz: Tuple[float, ...] = (0.17, 0.41, 0.83, 1.31)
    evasive_amps_m: Tuple[float, ...] = (45.0, 30.0, 14.0, 7.0)
    evasive_step_interval_s: float = 3.5
    evasive_step_m: float = 60.0

    #: Static bench target: the control case. If the tracker cannot hold a
    #: stationary beacon to well inside tolerance, nothing else it does counts.
    static_az_deg: float = 2.0
    static_el_deg: float = 15.0
    static_range_km: float = 10.0


# --------------------------------------------------------------------------- #
#  root
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class AppConfig:
    """Root configuration object. One per run, captured in the run log."""

    #: Master seed. Fixed by default: a demo that cannot be reproduced is an
    #: anecdote, and the performance log is only evidence if the run behind it
    #: can be repeated exactly.
    seed: int = 20260828

    camera: CameraConfig = field(default_factory=CameraConfig)
    turbulence: TurbulenceConfig = field(default_factory=TurbulenceConfig)
    jitter: JitterConfig = field(default_factory=JitterConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    ekf: EkfConfig = field(default_factory=EkfConfig)
    pid: PidConfig = field(default_factory=PidConfig)
    tuner: TunerConfig = field(default_factory=TunerConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    pat: PatConfig = field(default_factory=PatConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    disturbance: Disturbance = field(default_factory=Disturbance)

    #: Initial pointing error at t=0, radians (1.4 deg). Deliberately larger
    #: than the 1 deg FOV half-angle, so the beacon starts *outside the frame*
    #: and SEARCH has to genuinely find it. Starting with the target already
    #: visible would skip the acquisition problem the problem statement is
    #: actually about, and would make the acquisition-time metric meaningless.
    initial_offset_rad: float = 0.0244

    @property
    def dt(self) -> float:
        return 1.0 / self.pat.target_hz


DEFAULT_CONFIG = AppConfig()
