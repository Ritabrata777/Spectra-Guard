"""
SPECTRA GUARD -- sub-pixel centroid estimation.

A coarse tracker's angular precision is set by how well it can locate the beacon
*within* a pixel, not by the pixel size. At 54.5 urad/px, whole-pixel
quantisation alone would contribute 54.5/sqrt(12) = 15.7 urad RMS -- comparable
to the entire angle-of-arrival jitter budget. Sub-pixel estimation is therefore
not a refinement, it is the thing that makes the plate scale acceptable.

Two independent estimators are provided, and they fail in different ways, which
is why both exist.


1. Intensity-weighted centroid (first moment)
---------------------------------------------
Over a window W around the brightest pixel, with background b subtracted:

    u_hat = sum_W (I - b) * u  /  sum_W (I - b)
    v_hat = sum_W (I - b) * v  /  sum_W (I - b)

For a symmetric PSF in symmetric noise this is unbiased and, for a Gaussian of
width sigma at peak SNR rho, its variance approaches the Cramer-Rao-like result

    var(u_hat) ~ sigma^2 / rho^2       [px^2]

so precision improves linearly with SNR, not as its square root. At sigma = 2.9
px and rho = 30 that is 0.1 px = 5.3 urad, comfortably below the AoA budget.

*** THE BACKGROUND-BIAS TRAP ***

The first moment is only unbiased if the background is *exactly* removed. Any
residual pedestal r over a window of half-width h drags the estimate toward the
window centre, because the pedestal's own centroid IS the window centre:

    u_hat = (F * u_psf + R * u_centre) / (F + R)

where F is the beacon flux and R = r * (2h+1)^2 the residual pedestal flux. With
h = 5, a residual of just 1 DN gives R = 121 DN against a beacon flux of
2*pi*sigma^2*peak ~ 5900 DN, i.e. a 2% pull toward the window centre -- 0.1 px
for a beacon 5 px off centre. That is the same size as the entire noise-limited
precision, and it is *systematic*, so averaging does not remove it.

This is why the estimator below subtracts a background measured from an annulus
around the window rather than a global frame statistic, and why it clips the
background-subtracted window at zero. Both matter. A global background is wrong
under the scene's vertical sky gradient and cos^4 vignetting; a signed residual
biases as derived above, while clipping at zero makes the residual one-sided and
much smaller (it converts a bias into a small variance inflation).


2. Parabolic / Gaussian 1-D peak fit
------------------------------------
Fit a parabola through the log of the three samples straddling the peak. For a
Gaussian, ln(I) is exactly a parabola, so the vertex of the log-parabola is the
exact PSF centre in the noiseless case:

    delta = 0.5 * (ln I[-1] - ln I[+1]) / (ln I[-1] - 2 ln I[0] + ln I[+1])
    u_hat = u_peak + delta,     |delta| <= 0.5

This uses only 3 samples per axis, so it is nearly immune to the background
pedestal that plagues the moment method -- a constant added to a Gaussian is not
a Gaussian, but over 3 samples near the peak the distortion is second order.
Its weakness is the mirror image: it uses so few samples that it is noisier, and
it needs the peak sample to actually be the brightest, which a single hot pixel
can break.

Known bias: for undersampled spots (sigma < ~0.8 px) the 3-point log-parabola
develops an S-shaped error curve of amplitude up to ~0.05 px, peaking at
|delta| ~ 0.25 -- the classic "pixel-phase" error familiar from astrometry. At
our sigma = 1.9-7.0 px the spot is well sampled and this bias is below 0.005 px,
so it is not corrected. If the optics were ever changed to critically sample the
PSF, this comment is the place to start.


Which is used
-------------
``refine`` runs the moment estimator on a local, annulus-background-subtracted
window and uses the log-parabola as a **sanity check**: if the two disagree by
more than ``max_disagreement_px`` the window is contaminated (a neighbouring
star inside the window, or a cosmic-ray-like hot pixel), and the more robust
3-point fit is returned with ``subpixel=False`` to tell the EKF to inflate R.
Disagreement between two estimators with different failure modes is a free
integrity check; throwing it away would be wasteful.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class CentroidResult:
    """Sub-pixel centroid with a quality flag and its supporting diagnostics."""

    u: float
    v: float
    #: False when the two estimators disagreed, i.e. the window is contaminated
    #: and the result should be trusted less (the EKF inflates R accordingly).
    subpixel: bool
    #: Background-subtracted flux inside the window [DN]. Proxy for total signal.
    flux: float
    #: Peak DN above the local background.
    peak: float
    #: Second-moment width [px], sqrt of the mean of the two axis variances.
    sigma_px: float
    #: |moment - parabola| disagreement [px]. Reported for diagnostics.
    disagreement_px: float


def _log_parabola(a: float, b: float, c: float) -> float:
    """
    Vertex offset of the parabola through ln(a), ln(b), ln(c) at u = -1, 0, +1.

        delta = 0.5 * (ln a - ln c) / (ln a - 2 ln b + ln c)

    Returns 0.0 rather than raising when the fit is degenerate. Three guards:

      * any sample <= 0 (possible after background subtraction) has no log;
      * a non-negative denominator means the samples are not peaked, so there is
        no maximum to find and the vertex would be a *minimum* -- following it
        would push the estimate away from the beacon;
      * |delta| > 0.5 means the true peak is outside the bracketing samples, so
        the wrong pixel was chosen as the peak; clamping (rather than accepting)
        keeps the answer inside the pixel it belongs to.
    """
    if a <= 0.0 or b <= 0.0 or c <= 0.0:
        return 0.0
    la, lb, lc = math.log(a), math.log(b), math.log(c)
    denom = la - 2.0 * lb + lc
    if denom >= -1e-12:
        return 0.0
    delta = 0.5 * (la - lc) / denom
    if delta > 0.5:
        return 0.5
    if delta < -0.5:
        return -0.5
    return delta


def annulus_background(
    img: np.ndarray, u0: int, v0: int, r_in: int, r_out: int
) -> Tuple[float, float]:
    """
    Median and robust sigma of the background in an annulus about (u0, v0).

    Returns ``(median, sigma)`` where sigma is derived from the median absolute
    deviation:

        sigma ~ 1.4826 * MAD

    The **median**, not the mean, because the annulus will sometimes contain a
    star or a decoy wing, and a single bright outlier shifts a mean enough to
    matter (see the background-bias analysis in the module docstring). The MAD
    scaling constant 1.4826 = 1/Phi^-1(0.75) makes it a consistent estimator of
    a Gaussian's standard deviation.

    The annulus, not a disc, so the beacon's own wings are excluded: with
    r_in >= 3*sigma_psf the enclosed beacon flux is under 1.2%.
    """
    h, w = img.shape
    u_lo = max(0, u0 - r_out)
    u_hi = min(w, u0 + r_out + 1)
    v_lo = max(0, v0 - r_out)
    v_hi = min(h, v0 + r_out + 1)
    patch = img[v_lo:v_hi, u_lo:u_hi]
    if patch.size == 0:
        return 0.0, 1.0

    yy, xx = np.mgrid[v_lo:v_hi, u_lo:u_hi]
    r2 = (xx - u0) ** 2 + (yy - v0) ** 2
    mask = (r2 >= r_in * r_in) & (r2 <= r_out * r_out)
    vals = patch[mask]
    if vals.size < 8:
        # Too close to the frame edge for a meaningful annulus; fall back to the
        # whole patch. Biased high by the beacon, but a poor background is much
        # better than none, and the caller's zero-clip limits the damage.
        vals = patch.ravel()
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    return med, max(1.4826 * mad, 1e-3)


def refine(
    img: np.ndarray,
    u_guess: float,
    v_guess: float,
    half_window: int = 5,
    psf_sigma_hint: float = 2.0,
    max_disagreement_px: float = 1.25,
) -> Optional[CentroidResult]:
    """
    Sub-pixel refine a coarse (u, v) guess on a single-channel float image.

    ``img`` must be 2-D float (DN). ``u_guess``/``v_guess`` come from a
    connected-component centroid or a YOLO box centre and need only be accurate
    to within ``half_window``.

    Returns ``None`` when there is nothing there -- specifically when the peak
    does not exceed the local background by 3 robust sigma. Returning None
    rather than a low-confidence answer matters: a fabricated centroid on pure
    noise is worse than a missed frame, because the EKF would happily fuse it.
    """
    h, w = img.shape
    u_pk = int(round(u_guess))
    v_pk = int(round(v_guess))
    if not (0 <= u_pk < w and 0 <= v_pk < h):
        return None

    hw = int(max(2, half_window))
    # Background annulus sized from the PSF: inner radius at 3 sigma excludes
    # ~99% of the beacon flux, outer far enough out to collect enough samples.
    r_in = max(hw + 2, int(math.ceil(3.0 * psf_sigma_hint)) + 1)
    r_out = r_in + 6
    bg, bg_sigma = annulus_background(img, u_pk, v_pk, r_in, r_out)

    u_lo = max(0, u_pk - hw)
    u_hi = min(w, u_pk + hw + 1)
    v_lo = max(0, v_pk - hw)
    v_hi = min(h, v_pk + hw + 1)
    win = img[v_lo:v_hi, u_lo:u_hi].astype(np.float64) - bg

    # Re-locate the peak inside the window: the caller's guess is a blob
    # centroid, which for an asymmetric blob is not the brightest pixel.
    idx = int(np.argmax(win))
    pv, pu = np.unravel_index(idx, win.shape)
    peak = float(win[pv, pu])
    if peak < 3.0 * bg_sigma:
        return None

    # --- moment estimator on the zero-clipped window ---------------------- #
    # Clipping at zero rather than keeping signed residuals: a signed pedestal
    # produces a systematic pull toward the window centre (module docstring),
    # whereas clipping converts that bias into a small variance inflation.
    wclip = np.maximum(win, 0.0)
    total = float(wclip.sum())
    if total <= 1e-9:
        return None

    us = np.arange(u_lo, u_hi, dtype=np.float64)
    vs = np.arange(v_lo, v_hi, dtype=np.float64)
    wu = wclip.sum(axis=0)
    wv = wclip.sum(axis=1)
    u_mom = float((wu * us).sum() / total)
    v_mom = float((wv * vs).sum() / total)

    # Second moments -> apparent width, used for shape scoring upstream.
    var_u = float((wu * (us - u_mom) ** 2).sum() / total)
    var_v = float((wv * (vs - v_mom) ** 2).sum() / total)
    sigma_px = math.sqrt(max(0.5 * (var_u + var_v), 0.0))

    # --- 3-point log-parabola, as an independent cross-check -------------- #
    u_par = float(u_lo + pu)
    v_par = float(v_lo + pv)
    if 0 < pu < win.shape[1] - 1:
        u_par += _log_parabola(win[pv, pu - 1], peak, win[pv, pu + 1])
    if 0 < pv < win.shape[0] - 1:
        v_par += _log_parabola(win[pv - 1, pu], peak, win[pv + 1, pu])

    disagreement = math.hypot(u_mom - u_par, v_mom - v_par)
    if disagreement <= max_disagreement_px:
        return CentroidResult(
            u=u_mom, v=v_mom, subpixel=True, flux=total, peak=peak,
            sigma_px=sigma_px, disagreement_px=disagreement,
        )
    # Estimators disagree => the window is contaminated (neighbouring star, hot
    # pixel, or two merged blobs). Prefer the 3-point fit, which only looks at
    # the immediate neighbourhood of the peak, and flag reduced confidence.
    return CentroidResult(
        u=u_par, v=v_par, subpixel=False, flux=total, peak=peak,
        sigma_px=sigma_px, disagreement_px=disagreement,
    )


__all__ = ["CentroidResult", "annulus_background", "refine"]
