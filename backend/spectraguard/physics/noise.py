"""
SPECTRA GUARD -- focal-plane sensor noise.

Three physically distinct noise sources are modelled, because they scale
differently with signal and therefore behave differently as the operator dials
the beacon into a fade:

1. **Shot (photon) noise** -- signal dependent, Poisson. The arrival of photons
   is a Poisson point process, so a pixel collecting N photoelectrons has
   variance N. In DN (digital numbers) with conversion gain g [e-/DN],
   S_DN = N/g, so

       var_DN = N / g^2 = S_DN / g

   This is the *dominant* noise on a bright beacon and it is the reason a
   brighter beacon does not improve centroid SNR as fast as one might hope:
   signal grows as N, noise as sqrt(N), so SNR only grows as sqrt(N).

2. **Dark current** -- signal independent, also Poisson (thermally generated
   carriers are an independent Poisson process). Adds both a DC pedestal and
   its own shot noise.

3. **Read noise + operator AWGN** -- signal independent, Gaussian. Read noise is
   the sensor's electronics floor; ``awgn_sigma`` is the operator-injected
   additive noise that lets the jury watch the detector degrade. Independent
   Gaussians add in quadrature:

       sigma_gauss^2 = read_noise^2 + awgn_sigma^2

Total per-pixel variance:

    var_total(S) = (S_DN + dark_DN) / g  +  read^2 + awgn^2
                   \\______ signal dep. ______/   \\___ fixed ___/


Why the default path uses the Gaussian limit of the Poisson
-----------------------------------------------------------
Sampling an exact Poisson deviate per pixel costs ~12.7 ms for a 640x480 frame
with numpy's Generator on the reference machine. The frame budget at 60 Hz is
16.7 ms *in total*, including render, detect, estimate and control -- so exact
Poisson sampling alone would consume 76% of it and the loop could not hold rate.

The Gaussian limit costs ~2.3 ms because the signal-dependent and
signal-independent variances can be combined analytically and drawn with a
**single** ``standard_normal`` call:

    sigma_total(S) = sqrt( (S + dark)/g + read^2 + awgn^2 )
    out = S + dark + sigma_total(S) * n,     n ~ N(0,1)

Statistical justification. Poisson(lambda) -> Normal(lambda, lambda) with
relative skewness 1/sqrt(lambda) and excess kurtosis 1/lambda. At the signal
levels that matter here -- beacon peak ~112 DN at g = 4 e-/DN, i.e.
lambda ~ 450 e- -- skewness is 0.047 and excess kurtosis 0.0022. Both are far
below the level at which any moment-based centroid estimator could distinguish
the two distributions, and both are utterly negligible against the *systematic*
errors we actually care about (turbulent tilt, PSF broadening, background
gradient). Even in the sky background at ~26 DN, lambda ~ 104 e- gives skewness
0.098.

The approximation is only questionable in the deep-fade tail, where a scintilled
beacon drops to a few DN and lambda falls to O(10). Two mitigations: the exact
path is available via ``NoiseConfig.exact_poisson`` for offline validation, and
the Gaussian draw is clipped at zero (negative photon counts are unphysical and
would otherwise let the detector's top-hat see structure that cannot exist).

Set ``exact_poisson=True`` when producing numbers for a paper. Leave it off for
the live demo. Either way the *variance* -- which is what sets centroid
precision -- is identical by construction; only the third and higher moments
differ.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..config import NoiseConfig


class SensorNoise:
    """
    Applies dark current, shot noise and Gaussian read/AWGN to a float image.

    Operates on a float32 image in DN and returns a float32 image in DN. The
    caller is responsible for the final clip-and-cast to uint8, because the
    scene renderer wants to composite in float and quantise exactly once --
    quantising twice would add a spurious extra LSB of noise.
    """

    def __init__(self, cfg: NoiseConfig, rng: Optional[np.random.Generator] = None) -> None:
        self.cfg = cfg
        self._rng = rng if rng is not None else np.random.default_rng()

    def apply(self, image_dn: np.ndarray, awgn_sigma: float) -> np.ndarray:
        """
        Add the full noise stack to ``image_dn`` [DN, float32].

        ``awgn_sigma`` is the operator-injected Gaussian sigma in 8-bit DN; it
        adds in quadrature with the sensor's own read noise.
        """
        cfg = self.cfg
        img = np.asarray(image_dn, dtype=np.float32)

        # Dark current is a DC pedestal; its own shot noise is folded into the
        # signal-dependent term below by adding it to the Poisson rate.
        rate = img + np.float32(cfg.dark_current_dn)
        np.maximum(rate, 0.0, out=rate)

        gauss_var = float(cfg.read_noise_dn) ** 2 + float(max(awgn_sigma, 0.0)) ** 2

        if cfg.exact_poisson:
            # Reference path: true Poisson deviates. ~5.5x slower; see module
            # docstring for when this is worth paying for.
            lam_e = rate * np.float32(cfg.gain_e_per_dn)
            out = self._rng.poisson(lam_e).astype(np.float32) / np.float32(cfg.gain_e_per_dn)
            if gauss_var > 0.0:
                out += self._rng.standard_normal(out.shape, dtype=np.float32) * np.float32(
                    np.sqrt(gauss_var)
                )
        else:
            # Fast path: combine both variances, draw once.
            #   var = rate/g + read^2 + awgn^2
            var = rate / np.float32(cfg.gain_e_per_dn) + np.float32(gauss_var)
            sigma = np.sqrt(var, out=var)  # in-place: var is a fresh temporary
            noise = self._rng.standard_normal(img.shape, dtype=np.float32)
            out = rate + sigma * noise

        # Negative counts are unphysical. Leaving them in would let the
        # detector's top-hat filter respond to structure that no real sensor
        # could produce, which flatters the detector.
        np.maximum(out, 0.0, out=out)
        return out

    def snr(self, signal_dn: float, background_dn: float, awgn_sigma: float) -> float:
        """
        Single-pixel SNR of a signal on a background -- a reporting helper.

            SNR = S / sqrt( (S + B + dark)/g + read^2 + awgn^2 )

        Useful for sanity-checking that a configuration is even detectable
        before blaming the detector for missing it.
        """
        cfg = self.cfg
        var = (
            (max(signal_dn, 0.0) + max(background_dn, 0.0) + cfg.dark_current_dn)
            / cfg.gain_e_per_dn
            + cfg.read_noise_dn ** 2
            + max(awgn_sigma, 0.0) ** 2
        )
        return float(signal_dn / np.sqrt(var)) if var > 0 else 0.0


__all__ = ["SensorNoise"]
