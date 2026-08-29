"""
SPECTRA GUARD -- platform micro-vibration (jitter).

Why not white noise
-------------------
Platform jitter on a real optical terminal is emphatically **not** white. It is
the structural response of the airframe / bus / gimbal assembly to broadband
mechanical excitation (reaction wheels, cryocooler, rotor blade passage, engine
harmonics, aerodynamic buffet). The structure has modes, so the disturbance PSD
has *peaks*: a few narrow bands carrying most of the power.

This distinction decides whether the whole control problem is easy or hard:

  * White disturbance is trivially attenuated. Its power is spread over all
    frequencies, so only the fraction inside the loop bandwidth matters, and a
    modest integrator handles it.
  * Narrowband disturbance is not. If the peak sits near the loop's crossover
    frequency the controller *amplifies* it (that is what a resonant peak in the
    sensitivity function does), and if it sits above crossover the loop cannot
    see it at all. Either way, gain tuning has to know where the peak is.

That is exactly why ``psd_peak()`` exists and why the peak frequency is fed to
the gain tuner as an observation (see control/sac_tuner.py). A tuner blind to
the disturbance spectrum can only react to error it has already committed; one
that knows the peak frequency can pre-emptively trade proportional gain for
derivative gain. Modelling jitter as white noise would delete this entire part
of the problem and make the adaptive tuner look pointless.


Model
-----
A single dominant structural mode per axis, realised as a 2nd-order resonant
IIR (an RBJ biquad bandpass) driven by white Gaussian noise:

    H(z) = (b0 + b1 z^-1 + b2 z^-2) / (a0 + a1 z^-1 + a2 z^-2)

    w0    = 2*pi*f0/fs
    alpha = sin(w0) / (2*Q)

    b0 = +alpha,  b1 = 0,          b2 = -alpha
    a0 = 1+alpha, a1 = -2*cos(w0), a2 = 1-alpha

all divided through by a0. The pole pair sits at radius ~ (1-alpha)/(1+alpha)
and angle w0, giving a resonance at f0 with -3 dB bandwidth f0/Q. With Q = 8 the
output is clearly narrowband -- an obvious spectral peak -- without ringing so
long that the process stops looking random.

Direct Form II transposed is used for the recursion because it is the standard
numerically well-behaved arrangement for a biquad and keeps the state to two
scalars per axis.

Amplitude calibration
---------------------
The biquad's white-noise output RMS depends on f0, Q *and* fs in a way that has
a closed form but an ugly one. Rather than trust algebra that would silently
drift if the filter form changed, the realised RMS is *measured* once by pushing
a calibration burst of white noise through the filter, and an input gain is set
so the output RMS is exactly 1.0. The commanded ``jitter_urad`` then multiplies
a unit-variance process, so the telemetry number means what it says. The
calibration is cached and only recomputed when f0, Q or fs change.

The two axes use independent noise streams and independent filter states, so
pan and tilt jitter are uncorrelated -- appropriate for two orthogonal
structural modes that happen to share a resonant frequency.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np

from ..config import JitterConfig


@dataclass
class BiquadCoeffs:
    """Normalised biquad coefficients (a0 divided out)."""

    b0: float
    b1: float
    b2: float
    a1: float
    a2: float


def design_bandpass(f0_hz: float, q: float, fs_hz: float) -> BiquadCoeffs:
    """
    RBJ constant-peak-gain bandpass biquad.

    ``f0_hz`` is clamped to (0, 0.45*fs). Above Nyquist the design is
    meaningless, and an operator dragging the jitter-frequency slider past it
    must degrade gracefully rather than produce NaNs or an unstable pole pair.
    """
    fs_hz = max(float(fs_hz), 1.0)
    f0 = min(max(float(f0_hz), 1e-3), 0.45 * fs_hz)
    q = max(float(q), 0.5)

    w0 = 2.0 * math.pi * f0 / fs_hz
    alpha = math.sin(w0) / (2.0 * q)
    a0 = 1.0 + alpha
    return BiquadCoeffs(
        b0=alpha / a0,
        b1=0.0,
        b2=-alpha / a0,
        a1=(-2.0 * math.cos(w0)) / a0,
        a2=(1.0 - alpha) / a0,
    )


class _Biquad:
    """Single-channel Direct Form II transposed biquad."""

    __slots__ = ("c", "_z1", "_z2")

    def __init__(self, c: BiquadCoeffs) -> None:
        self.c = c
        self._z1 = 0.0
        self._z2 = 0.0

    def reset(self) -> None:
        self._z1 = 0.0
        self._z2 = 0.0

    def step(self, x: float) -> float:
        c = self.c
        y = c.b0 * x + self._z1
        self._z1 = c.b1 * x - c.a1 * y + self._z2
        self._z2 = c.b2 * x - c.a2 * y
        return y

    def run(self, x: np.ndarray) -> np.ndarray:
        """
        Filter a block, vectorised where possible.

        The recursion is inherently sequential, so this is a Python loop -- but
        it is only ever used on the one-off calibration burst, never in the
        per-frame path, where ``step`` handles one sample per axis per frame.
        """
        out = np.empty_like(x)
        for i, xi in enumerate(x):
            out[i] = self.step(float(xi))
        return out


@dataclass
class JitterSample:
    """One frame of 2-axis angular disturbance [rad]."""

    pan_rad: float
    tilt_rad: float


class PlatformJitter:
    """
    Narrowband 2-axis platform vibration generator.

    ``step()`` returns the instantaneous angular disturbance to be *added to the
    achieved gimbal pointing* -- it is a disturbance on the plant output, not on
    the command, because a structural mode shakes the whole optical head
    regardless of what the rate loop is doing.
    """

    def __init__(
        self,
        cfg: JitterConfig,
        fs_hz: float,
        jitter_hz: float = 12.0,
        jitter_urad: float = 40.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.cfg = cfg
        self.fs = float(fs_hz)
        self._rng = rng if rng is not None else np.random.default_rng()

        self._f0 = float(jitter_hz)
        self._amp_urad = float(jitter_urad)
        self._q = float(cfg.q)

        self._bq_pan = _Biquad(design_bandpass(self._f0, self._q, self.fs))
        self._bq_tilt = _Biquad(design_bandpass(self._f0, self._q, self.fs))
        self._input_gain = 1.0
        self._calibrate()

        # History of the pan-axis disturbance, in microradians, for the PSD.
        self._hist: Deque[float] = deque(maxlen=int(cfg.psd_history))
        self._last = JitterSample(0.0, 0.0)

    # ---------------------------------------------------------- calibration
    def _calibrate(self) -> None:
        """
        Measure the biquad's unit-white-noise output RMS and invert it.

        Uses a throwaway filter instance so the live filter states are not
        disturbed, and discards the first 10% of the burst as transient -- the
        biquad starts from rest, and including its settling would bias the RMS
        low, making the delivered jitter amplitude too large.
        """
        n = max(1024, int(self.cfg.calibration_samples))
        probe = _Biquad(self._bq_pan.c)
        x = self._rng.standard_normal(n)
        y = probe.run(x)
        y = y[n // 10 :]
        rms = float(np.sqrt(np.mean(y * y)))
        self._input_gain = 1.0 / rms if rms > 1e-12 else 0.0

    # ------------------------------------------------------------ parameters
    def set_params(self, jitter_hz: Optional[float] = None, jitter_urad: Optional[float] = None) -> None:
        """
        Retune live. Only redesigns (and recalibrates) if f0 actually moved.

        Amplitude changes are free -- they are a post-multiply on a
        unit-variance process -- so dragging the amplitude slider never costs a
        recalibration, which matters when the UI sends patches at 20 Hz.
        """
        if jitter_urad is not None:
            self._amp_urad = float(jitter_urad)
        if jitter_hz is not None and abs(float(jitter_hz) - self._f0) > 1e-9:
            self._f0 = float(jitter_hz)
            c = design_bandpass(self._f0, self._q, self.fs)
            # Keep the existing filter states: replacing coefficients without
            # resetting gives a smooth spectral slide rather than a click, which
            # is what a real structure does as its loading changes.
            self._bq_pan.c = c
            self._bq_tilt.c = c
            self._calibrate()

    @property
    def jitter_hz(self) -> float:
        return self._f0

    @property
    def jitter_urad(self) -> float:
        return self._amp_urad

    # ------------------------------------------------------------------ step
    def step(self) -> JitterSample:
        """Advance one frame and return the 2-axis disturbance in radians."""
        scale = self._amp_urad * 1e-6 * self._input_gain
        p = self._bq_pan.step(float(self._rng.standard_normal())) * scale
        t = self._bq_tilt.step(float(self._rng.standard_normal())) * scale
        self._hist.append(p * 1e6)
        self._last = JitterSample(p, t)
        return self._last

    @property
    def last(self) -> JitterSample:
        return self._last

    def amplitude_rms_rad(self) -> float:
        """Commanded 1-axis RMS disturbance [rad]."""
        return self._amp_urad * 1e-6

    # ------------------------------------------------------------------- PSD
    def psd_peak(self) -> Tuple[float, float]:
        """
        Locate the dominant spectral peak in the recent jitter history.

        Returns ``(peak_frequency_hz, peak_power)``. This is the tuner's
        observation of platform state (control/sac_tuner.py): the controller is
        not told ``jitter_hz`` directly, it has to *estimate* it from the
        disturbance it observes, exactly as a real system would.

        Uses ``scipy.signal.welch`` when SciPy is importable and an equivalent
        pure-numpy Welch estimator otherwise. SciPy is an optional dependency
        here on purpose -- the fallback is ~30 lines and means the backend runs
        on a bare numpy install.
        """
        n = len(self._hist)
        if n < 32:
            return self._f0, 0.0
        x = np.fromiter(self._hist, dtype=np.float64, count=n)
        x = x - x.mean()
        nperseg = min(n, 128)
        try:
            from scipy import signal as _sig  # noqa: PLC0415 -- optional dep

            freqs, pxx = _sig.welch(x, fs=self.fs, nperseg=nperseg)
        except Exception:
            freqs, pxx = _welch_numpy(x, self.fs, nperseg)
        if freqs.size < 2:
            return self._f0, 0.0
        # Skip the DC bin: a residual mean would otherwise always win.
        k = int(np.argmax(pxx[1:])) + 1
        return float(freqs[k]), float(pxx[k])


def _welch_numpy(x: np.ndarray, fs: float, nperseg: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Welch PSD estimate using only numpy: Hann window, 50% overlap.

    Segment-averaging trades frequency resolution for variance reduction. A
    single periodogram has ~100% standard error at every bin regardless of
    record length, which is far too noisy to pick a peak from reliably;
    averaging K half-overlapped segments cuts that by roughly sqrt(K).

    Normalisation matches ``scipy.signal.welch(..., scaling='density')``:

        Pxx = |FFT(w * x)|^2 / (fs * sum(w^2))

    with the one-sided spectrum doubled except at DC and Nyquist. The absolute
    scaling does not matter for locating the peak, but matching SciPy means the
    two code paths are interchangeable and the fallback can be validated
    against the reference implementation.
    """
    nperseg = int(min(max(nperseg, 8), x.size))
    step = max(1, nperseg // 2)
    win = np.hanning(nperseg)
    win_pow = float(np.sum(win * win))

    starts = range(0, x.size - nperseg + 1, step)
    acc = None
    count = 0
    for s in starts:
        seg = x[s : s + nperseg] * win
        spec = np.abs(np.fft.rfft(seg)) ** 2
        acc = spec if acc is None else acc + spec
        count += 1
    if acc is None or count == 0:
        return np.array([0.0]), np.array([0.0])

    pxx = acc / count / (fs * win_pow)
    # One-sided: fold in the negative frequencies, except DC and (if present)
    # the Nyquist bin, which have no mirror partner.
    if pxx.size > 2:
        pxx[1:-1] *= 2.0
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    return freqs, pxx


__all__ = [
    "BiquadCoeffs",
    "design_bandpass",
    "JitterSample",
    "PlatformJitter",
]
