"""
SPECTRA GUARD -- synthetic focal-plane image generation.

Renders a 640x480 uint8 BGR frame per tick, fully vectorised. This is the
detector's only view of the world, so every confuser we omit is a confuser our
detector never has to beat. The scene therefore includes, on purpose, several
things that make life harder:

  * a **sky/space background gradient**, so a fixed global threshold cannot work
    and the detector has to remove the background;
  * a **star field**, giving many small bright point sources -- structurally
    identical to the beacon;
  * **cos^4 vignetting**, so the same target is dimmer at the frame edge than at
    the centre and absolute-brightness rules break;
  * **2-3 bright decoys**, at least one *brighter than the beacon*, so a
    "brightest pixel" or "largest blob" tracker demonstrably fails and the
    detector's prior-gated scoring has to earn its keep;
  * a **moving cloud** that genuinely extinguishes the beacon, forcing COAST
    rather than merely dimming it;
  * **turbulence**: centroid displaced by the frozen-flow tilt, peak modulated
    by log-normal scintillation, PSF broadened as r0 shrinks;
  * **shot + read + operator noise** applied last, in that order.

Beacon PSF
----------
A 2-D Gaussian approximating the Airy core. The Airy pattern's central lobe is
matched well by a Gaussian with sigma ~ 0.42 * lambda*F/# (in linear measure);
the rings carry only ~16% of the energy and are far below the noise floor here,
so the Gaussian is entirely adequate for centroiding studies.

    I(u,v) = A * exp( -[(u-u0)^2 + (v-v0)^2] / (2*sigma^2) )

with

    sigma = sigma_diffraction * broadening,   broadening = max(1, D/r0)^(5/6)
    A     = A0 * scint * (sigma_0/sigma)^2

The (sigma_0/sigma)^2 factor **conserves flux**: turbulence redistributes energy
across a wider spot, it does not absorb it. This is why the reported
``scintillation`` attenuation can be small at high Cn2 without the total signal
vanishing -- and it is why the beacon becomes hard to centroid (low peak SNR,
broad spot) rather than simply invisible.

Performance
-----------
The frame budget at 60 Hz is 16.7 ms for render + detect + estimate + control,
so the render has ~6 ms. Three decisions get it there:

1. **The static background is precomputed once.** Gradient, stars and vignetting
   depend only on geometry, not on time, so they are baked into one float32
   layer at construction and copied per frame. A per-frame star field would cost
   more than everything else combined.
2. **Point sources are rendered into small local windows**, not full-frame. A
   full-frame ``exp()`` is ~3 ms *per source*; a 25x25 window is ~2 us. With a
   beacon plus 3 decoys that is the difference between 12 ms and 8 us.
3. **The cloud is synthesised on a 60x80 coarse grid and upsampled** with
   ``cv2.resize(INTER_CUBIC)``. A cloud is by nature low-spatial-frequency, so
   computing it at full resolution buys nothing; the coarse path is ~0.95 ms
   against ~10 ms for full-resolution smoothing.

There are no per-pixel Python loops anywhere in this file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from ..config import CameraConfig, Disturbance, SceneConfig
from ..physics.noise import SensorNoise
from ..physics.turbulence import TurbulenceSample
from .camera import PinholeCamera


@dataclass
class SceneRender:
    """One rendered frame plus the ground truth that produced it."""

    frame: np.ndarray  # uint8 BGR (H, W, 3)
    beacon_u: float
    beacon_v: float
    #: True iff the beacon is inside the frame AND not extinguished by cloud.
    #: This is the ground-truth answer to "should the detector see it?", and is
    #: what the engine's COAST logic is scored against.
    beacon_visible: bool
    #: Multiplicative log-normal irradiance factor applied this frame.
    scint_factor: float
    #: Reported 0..1 peak-intensity attenuation vs an ideal unscintillated,
    #: diffraction-limited beacon. Goes on the wire as `scintillation`.
    attenuation: float
    #: Cloud alpha at the beacon location, 0..1.
    occlusion_alpha: float
    #: Rendered PSF sigma [px] after turbulent broadening.
    psf_sigma_px: float
    #: Beacon peak DN actually written into the frame, before noise.
    peak_dn: float


class SceneRenderer:
    """
    Vectorised focal-plane renderer.

    One instance per run. Holds the precomputed static background, the decoy
    catalogue and the sensor-noise model.
    """

    def __init__(
        self,
        scene: SceneConfig,
        camera: CameraConfig,
        noise: SensorNoise,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.cfg = scene
        self.cam_cfg = camera
        self.noise = noise
        self._rng = rng if rng is not None else np.random.default_rng()
        self.w = camera.width
        self.h = camera.height

        self._background = self._build_background()
        self._decoys = self._build_decoys()
        # Scratch buffer reused every frame to avoid a 1.2 MB allocation at
        # 60 Hz, which would otherwise churn the allocator and the GC.
        self._buf = np.empty((self.h, self.w), dtype=np.float32)

    # ------------------------------------------------------------ background
    def _build_background(self) -> np.ndarray:
        """
        Precompute sky gradient + star field + vignetting as one float32 layer.

        Called once. Everything here is time-invariant, which is what makes the
        60 Hz budget achievable (see module docstring).
        """
        cfg = self.cfg
        h, w = self.h, self.w

        # Vertical sky gradient. Space/sky is brighter near the horizon.
        top, bot = cfg.sky_dn
        col = np.linspace(top, bot, h, dtype=np.float32)[:, None]
        bg = np.repeat(col, w, axis=1)

        # Star field, fixed seed so the same stars appear every run and a
        # reviewer can compare frames across runs. Stars are rendered as small
        # Gaussians, not single pixels: a single-pixel star is trivially
        # rejected by any shape test, which would make the clutter toothless.
        star_rng = np.random.default_rng(0xC0FFEE)
        n = int(cfg.n_stars)
        su = star_rng.uniform(0, w, n)
        sv = star_rng.uniform(0, h, n)
        lo, hi = cfg.star_peak_dn
        amp = star_rng.uniform(lo, hi, n)
        sig = star_rng.uniform(0.8, 1.5, n)
        for i in range(n):
            _add_gaussian(bg, float(su[i]), float(sv[i]), float(amp[i]), float(sig[i]))

        # cos^4 vignetting about the optical axis. Expressed via the off-axis
        # field angle, so it scales correctly with the configured FOV rather
        # than being an arbitrary radial falloff.
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        f = float(self.cam_cfg.f_px)
        r2 = (xx - w / 2.0) ** 2 + (yy - h / 2.0) ** 2
        cos_theta = f / np.sqrt(r2 + f * f)
        vig = cos_theta ** 4
        vig = 1.0 - cfg.vignette_strength * (1.0 - vig)
        self._vignette = vig.astype(np.float32)

        return (bg * self._vignette).astype(np.float32)

    def _build_decoys(self) -> np.ndarray:
        """
        Decoy catalogue: shape (n, 6) = (u, v, peak_frac, sigma, phase, twinkle_hz).

        Decoys are placed away from the frame centre so they do not sit on the
        boresight at t=0, and given a slow drift so they are not at fixed pixel
        coordinates a detector could memorise. Sigmas bracket the beacon's own
        so that shape alone cannot separate them -- the discriminators have to be
        the modulation tag and the EKF prior.

        ``twinkle_hz`` is drawn from 0.2-3.5 Hz, the band where atmospheric
        scintillation of a point source actually lives (the correlation time is
        tau_c = D/v = 40 ms, so power rolls off above ~4 Hz). This matters for
        fairness in both directions: the decoys must twinkle, or the modulation
        demodulator would be discriminating "varies at all" rather than "varies
        at 7 Hz"; but they must twinkle in the physically correct band rather
        than being handed a frequency adjacent to the tag, which would be
        adversarial rather than realistic.
        """
        cfg = self.cfg
        n = int(cfg.n_decoys)
        rng = np.random.default_rng(0xDEC01)
        out = np.zeros((n, 6), dtype=np.float64)
        for i in range(n):
            ang = 2.0 * math.pi * (i + 0.5) / n
            rad = rng.uniform(0.22, 0.40) * min(self.w, self.h)
            out[i, 0] = self.w / 2.0 + rad * math.cos(ang)
            out[i, 1] = self.h / 2.0 + rad * math.sin(ang)
            frac = cfg.decoy_peak_frac[i % len(cfg.decoy_peak_frac)]
            out[i, 2] = frac
            out[i, 3] = rng.uniform(0.85, 1.35) * cfg.beacon_sigma_px
            out[i, 4] = rng.uniform(0.0, 2.0 * math.pi)
            out[i, 5] = rng.uniform(0.2, 3.5)
        return out

    # ----------------------------------------------------------------- cloud
    def _cloud_alpha(self, t: float, occlusion: float) -> Optional[np.ndarray]:
        """
        Smooth moving cloud alpha field, or None when occlusion is off.

        Built as a sum of three drifting 2-D Gaussian blobs on a 60x80 grid,
        then bicubically upsampled. The blobs move at mutually incommensurate
        speeds so the coverage pattern does not simply translate, which would
        let the beacon pop back into view at a predictable period.

        Interpretation of the ``occlusion`` parameter
        --------------------------------------------
        ``occlusion`` is the fraction of sky covered by *optically thick* cloud,
        not a peak opacity. That is the physically right reading: water cloud at
        1550 nm attenuates by tens of dB over any depth you can see, so a cloud
        you can see at all is effectively opaque -- what varies with weather is
        how much of the sky it covers, not how grey it is.

        So the smooth blob density field ``s(x, y, t) in [0, 1]`` is *thresholded*
        rather than scaled:

            alpha = clamp( (s - (1 - occlusion)) / edge, 0, 1 )

        with ``edge = cloud_edge_softness``. At occlusion = 0 the threshold is 1
        and alpha is identically 0 (clear sky). At occlusion = 0.4 only the blob
        cores are opaque, so the beacon is lost briefly and intermittently. At
        occlusion = 1 the threshold is 0 and the sky is overcast.

        *** WHY IT IS DONE THIS WAY -- A BUG THIS REPLACED ***

        The first version multiplied the blob field by ``occlusion`` instead of
        thresholding it, so the *peak* alpha equalled ``occlusion``. Since the
        beacon is only declared lost above ``occlusion_lost_alpha = 0.62``, the
        entire lower two-thirds of the operator's slider range did literally
        nothing, and even at 0.85 only the exact blob centre was opaque enough
        to matter. Combined with a vertical-centre expression whose operator
        precedence made ``sy * speed * t % gh`` wrap by a full frame height at
        unpredictable times, the measured result was **1 occluded frame in
        300 at occlusion = 0.85** -- the scenario the COAST logic exists for was
        effectively never exercised. Thresholding makes coverage monotone and
        legible in the slider, and the vertical motion below is a bounded
        sinusoid with no modulo in it.
        """
        if occlusion <= 1e-3:
            return None
        cfg = self.cfg
        gh, gw = cfg.cloud_grid
        yy, xx = np.mgrid[0:gh, 0:gw].astype(np.float32)

        # Grid units per second, from the requested image-plane speed.
        speed = cfg.cloud_speed_px_s / float(self.w) * gw
        s = cfg.cloud_blob_sigma_frac * gw
        # Wrap with a margin of 3 sigma each side so a blob fully leaves the
        # field before re-entering; wrapping at the frame edge would make blobs
        # visibly teleport.
        margin = 3.0 * s
        span = gw + 2.0 * margin

        acc = np.zeros((gh, gw), dtype=np.float32)
        # (speed factor, vertical amplitude frac, vertical rate, phase offset).
        # The speed factors are mutually irrational-ish so the three blobs never
        # re-align into a periodic pattern over a demo-length run.
        for sx, v_amp, v_rate, ph in (
            (1.00, 0.24, 0.21, 0.00),
            (0.63, 0.31, 0.13, 0.37),
            (1.37, 0.18, 0.29, 0.71),
        ):
            cx = (sx * speed * t + ph * span) % span - margin
            # Bounded sinusoid -- no modulo, so cy is continuous in t.
            cy = gh * 0.5 + math.sin(2.0 * math.pi * v_rate * t + ph * 6.28) * gh * v_amp
            acc += np.exp(
                -(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * s * s))
            ).astype(np.float32)

        # Density field in [0, 1], then the coverage threshold described above.
        dens = np.clip(acc, 0.0, 1.0)
        thresh = 1.0 - float(occlusion)
        alpha = (dens - np.float32(thresh)) / np.float32(max(cfg.cloud_edge_softness, 1e-3))
        np.clip(alpha, 0.0, 1.0, out=alpha)
        out = cv2.resize(alpha, (self.w, self.h), interpolation=cv2.INTER_CUBIC)
        # Bicubic interpolation is not monotone -- its kernel has negative lobes,
        # so it overshoots at sharp edges. Measured max was 1.021, and an alpha
        # above 1 would make ``img *= (1 - alpha)`` go NEGATIVE, injecting a dark
        # halo around every cloud edge. Clip after resizing, not before.
        np.clip(out, 0.0, 1.0, out=out)
        return out

    # ---------------------------------------------------------------- render
    def render(
        self,
        t: float,
        beacon_uv: Tuple[float, float],
        turb: TurbulenceSample,
        dist: Disturbance,
        camera: PinholeCamera,
    ) -> SceneRender:
        """
        Render one frame.

        ``beacon_uv`` is the geometric (turbulence-free) beacon pixel position
        from ``PinholeCamera.project``; the turbulent tilt is applied here so
        that the atmosphere is a property of the scene rather than of the
        projection.
        """
        cfg = self.cfg
        f_px = camera.f_px

        # --- turbulent image motion --------------------------------------- #
        # Angle-of-arrival tilt -> focal-plane displacement, sigma_px = f*theta.
        du = turb.tilt_x_rad * f_px
        dv = -turb.tilt_y_rad * f_px  # +elevation is -v
        bu = beacon_uv[0] + du
        bv = beacon_uv[1] + dv

        # --- PSF broadening and flux-conserving peak ---------------------- #
        broaden = max(1.0, turb.d_over_r0) ** (5.0 / 6.0)
        sigma = min(cfg.beacon_sigma_px * broaden, cfg.beacon_sigma_max_px)
        flux_ratio = (cfg.beacon_sigma_px / sigma) ** 2
        # Amplitude-modulation tag. The beacon is deliberately blinked at a known
        # frequency so the receiver can tell it from a star; see
        # SceneConfig.beacon_mod_hz for why 7 Hz specifically.
        mod = 1.0 + cfg.beacon_mod_depth * math.sin(2.0 * math.pi * cfg.beacon_mod_hz * t)
        peak = cfg.beacon_peak_dn * turb.scintillation * flux_ratio * mod

        # --- compose ------------------------------------------------------ #
        img = self._buf
        np.copyto(img, self._background)

        # Decoys drift slowly on small circles.
        for i in range(self._decoys.shape[0]):
            du_, dv_, frac, dsig, ph, tw_hz = self._decoys[i]
            wob = 9.0
            cu = du_ + wob * math.cos(0.13 * t + ph)
            cv_ = dv_ + wob * math.sin(0.11 * t + ph)
            # Decoys are stars: they twinkle a little but are NOT scintillated
            # by our link's path, and they are not turbulence-broadened by the
            # same amount because they are at a different angle. Keeping them
            # distinct from the beacon's photometry is what makes the scene
            # honest -- if decoys tracked the beacon's brightness exactly, the
            # detector could separate them by correlation alone.
            #
            # Each decoy uses its OWN drawn twinkle frequency (0.2-3.5 Hz) and
            # its own phase. A single shared rate here would make every decoy
            # brighten and dim in lockstep, which is both unphysical and quietly
            # unfair to the detector in our favour: a demodulator could then
            # reject decoys by their mutual correlation rather than by actually
            # discriminating the beacon's 7 Hz tag.
            twinkle = 1.0 + 0.12 * math.sin(2.0 * math.pi * float(tw_hz) * t + 3.0 * ph)
            _add_gaussian(
                img,
                cu,
                cv_,
                cfg.beacon_peak_dn * float(frac) * twinkle * flux_ratio,
                float(dsig) * broaden,
            )

        vis_in_frame = camera.in_frame(bu, bv, margin=3.0 * sigma)
        if vis_in_frame:
            _add_gaussian(img, bu, bv, peak, sigma)

        # --- cloud occlusion (before noise, so the loss is real) ---------- #
        alpha_at_beacon = 0.0
        alpha = self._cloud_alpha(t, dist.occlusion)
        if alpha is not None:
            # Cloud both attenuates the scene and adds its own scattered
            # radiance -- a real cloud is bright, not black. Without the haze
            # term the "occluded" region would be darker than the sky and the
            # detector could trivially infer where the target was hiding.
            haze = 34.0
            img *= (1.0 - alpha)
            img += alpha * haze * self._vignette
            if 0 <= int(bv) < self.h and 0 <= int(bu) < self.w:
                alpha_at_beacon = float(alpha[int(bv), int(bu)])

        beacon_visible = bool(vis_in_frame and alpha_at_beacon < cfg.occlusion_lost_alpha)

        # --- sensor noise, last ------------------------------------------- #
        noisy = self.noise.apply(img, dist.awgn_sigma)

        frame = np.empty((self.h, self.w, 3), dtype=np.uint8)
        gray = np.clip(noisy, 0.0, 255.0).astype(np.uint8)
        # A monochrome SWIR focal plane replicated across BGR. Kept as a 3-plane
        # image because the JPEG transport and the frontend canvas both expect
        # colour, and because a YOLO backend would expect 3 channels.
        frame[:, :, 0] = gray
        frame[:, :, 1] = gray
        frame[:, :, 2] = gray

        attenuation = float(
            np.clip(turb.scintillation * flux_ratio * (1.0 - alpha_at_beacon), 0.0, 1.0)
        )
        return SceneRender(
            frame=frame,
            beacon_u=float(bu),
            beacon_v=float(bv),
            beacon_visible=beacon_visible,
            scint_factor=float(turb.scintillation),
            attenuation=attenuation,
            occlusion_alpha=alpha_at_beacon,
            psf_sigma_px=float(sigma),
            peak_dn=float(peak),
        )


def _add_gaussian(img: np.ndarray, u0: float, v0: float, amp: float, sigma: float) -> None:
    """
    Add a 2-D Gaussian to ``img`` in place, over a local window only.

        I += amp * exp(-((u-u0)^2 + (v-v0)^2) / (2 sigma^2))

    The window is +/-3.5 sigma, which captures 99.96% of a Gaussian's peak
    profile in each axis -- the truncation is far below one DN for any amplitude
    we use, and it turns an O(W*H) full-frame exponential into an O(sigma^2)
    one. This is the single most important optimisation in the renderer.

    Separability is exploited too: exp(-(x^2+y^2)/2s^2) = exp(-x^2/2s^2) *
    exp(-y^2/2s^2), so the 2-D patch is an outer product of two 1-D exponentials
    and only 2*w exponentials are evaluated instead of w^2.
    """
    if amp <= 0.0 or sigma <= 0.0:
        return
    h, w = img.shape
    rad = int(math.ceil(3.5 * sigma))
    u_lo = max(0, int(math.floor(u0)) - rad)
    u_hi = min(w, int(math.floor(u0)) + rad + 1)
    v_lo = max(0, int(math.floor(v0)) - rad)
    v_hi = min(h, int(math.floor(v0)) + rad + 1)
    if u_lo >= u_hi or v_lo >= v_hi:
        return

    inv = 1.0 / (2.0 * sigma * sigma)
    xs = np.arange(u_lo, u_hi, dtype=np.float32) - np.float32(u0)
    ys = np.arange(v_lo, v_hi, dtype=np.float32) - np.float32(v0)
    ex = np.exp(-(xs * xs) * inv)
    ey = np.exp(-(ys * ys) * inv)
    img[v_lo:v_hi, u_lo:u_hi] += np.float32(amp) * ey[:, None] * ex[None, :]


__all__ = ["SceneRender", "SceneRenderer"]
