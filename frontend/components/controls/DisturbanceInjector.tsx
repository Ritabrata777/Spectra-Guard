'use client';

/**
 * SPECTRA GUARD — Disturbance Injector.
 *
 * A floating action pill that morphs into the channel-conditions console. This is
 * the panel that wins the demo, because it is the one the evaluator gets to touch:
 * they drag Cn² into the strong-turbulence regime, watch the beacon fade and the
 * detector starve, and see the EKF hold pointing through it. Nothing we could say
 * about the algorithm lands as hard as letting them break it themselves.
 *
 * MORPH, DON'T SWAP. The pill and the expanded card are the same DOM element with
 * `layout` on it, so Framer Motion interpolates the geometry. A crossfade between
 * two separate elements would read as two objects; a single morphing surface reads
 * as one physical thing being opened, which is the entire premise of the material.
 *
 * The physics hints under each slider are load-bearing. They are what turn a set
 * of sliders into a statement that we know what the parameters mean.
 */

import { AnimatePresence, motion } from 'framer-motion';
import { useMemo } from 'react';
import { TactileSlider } from './TactileSlider';
import { useCommandStore } from '@/lib/store';
import { sci } from '@/lib/format';
import { DEFAULT_DISTURBANCE, type Disturbance } from '@/lib/types';

/**
 * Named atmospheric regimes.
 *
 * Cn² values are the standard ones from the FSOC literature: ~1e-17 for
 * astronomical seeing at a good site, ~1e-15 as a daytime horizontal-path
 * average, ~1e-13 for strong near-ground turbulence over hot terrain — which is
 * precisely the regime a vehicle-mounted terminal has to survive.
 */
const PRESETS: { id: string; label: string; note: string; patch: Disturbance }[] = [
  {
    id: 'calm',
    label: 'Benign',
    note: 'night, over water',
    patch: { cn2: 5e-17, jitterHz: 6, jitterUrad: 120, awgnSigma: 2, occlusion: 0 },
  },
  {
    id: 'nominal',
    label: 'Nominal',
    note: 'daytime horizontal path',
    patch: { ...DEFAULT_DISTURBANCE },
  },
  {
    id: 'severe',
    label: 'Severe',
    note: 'hot terrain, vehicle mount',
    patch: { cn2: 8e-14, jitterHz: 34, jitterUrad: 1600, awgnSigma: 11, occlusion: 0.15 },
  },
  {
    id: 'occlusion',
    label: 'Cloud transit',
    note: 'the coast-through case',
    patch: { cn2: 2e-14, jitterHz: 18, jitterUrad: 900, awgnSigma: 7, occlusion: 0.85 },
  },
];

/**
 * Collapse the five axes into one 0..1 severity figure for the pill label.
 *
 * Weighted, not averaged: occlusion and jitter dominate coarse-stage pointing
 * failure, while AWGN mostly costs centroid precision. Cn² enters through its
 * log because that is the axis it actually varies on.
 */
function severityOf(d: Disturbance): number {
  const cn2Norm = (Math.log10(d.cn2) + 17) / 4; // 1e-17..1e-13 -> 0..1
  const jitterNorm = Math.min(1, d.jitterUrad / 2000);
  const noiseNorm = Math.min(1, d.awgnSigma / 20);
  return Math.max(
    0,
    Math.min(1, 0.3 * cn2Norm + 0.32 * jitterNorm + 0.14 * noiseNorm + 0.24 * d.occlusion),
  );
}

function severityLabel(s: number): string {
  if (s < 0.25) return 'benign';
  if (s < 0.5) return 'nominal';
  if (s < 0.75) return 'degraded';
  return 'severe';
}

export function DisturbanceInjector() {
  const open = useCommandStore((s) => s.injectorOpen);
  const setOpen = useCommandStore((s) => s.setInjectorOpen);
  const d = useCommandStore((s) => s.disturbance);
  const setDisturbance = useCommandStore((s) => s.setDisturbance);
  const resetDisturbance = useCommandStore((s) => s.resetDisturbance);

  const severity = useMemo(() => severityOf(d), [d]);
  const sev = severityLabel(severity);
  const rgb = severity >= 0.75 ? '255,59,92' : severity >= 0.5 ? '255,169,43' : '34,224,255';

  const cn2 = sci(d.cn2);

  return (
    <div className="pointer-events-none fixed bottom-5 right-5 z-float flex flex-col items-end">
      <motion.div
        layout
        transition={{ layout: { duration: 0.5, ease: [0.32, 0.72, 0, 1] } }}
        className={[
          'glass glass-abyssal specular caustic pointer-events-auto overflow-hidden shadow-pill',
          open ? 'w-[21rem] rounded-glass' : 'rounded-pill',
        ].join(' ')}
        style={{ borderColor: `rgba(${rgb},0.4)` }}
      >
        {/* ------------------------------ handle ----------------------------- */}
        <motion.button
          layout="position"
          type="button"
          onClick={() => setOpen(!open)}
          aria-expanded={open}
          className="flex w-full items-center gap-3 px-4 py-2.5 text-left transition-colors duration-300 hover:bg-white/[0.04]"
        >
          <span
            className="relative flex h-2 w-2 shrink-0 items-center justify-center rounded-full"
            style={{
              backgroundColor: `rgb(${rgb})`,
              boxShadow: `0 0 10px rgba(${rgb},0.9)`,
            }}
          >
            {severity >= 0.5 && (
              <span
                className="absolute inset-0 animate-breathe rounded-full"
                style={{ boxShadow: `0 0 16px 4px rgba(${rgb},0.5)` }}
              />
            )}
          </span>

          <span className="flex flex-col">
            <span className="eyebrow">Disturbance injector</span>
            <span
              className="font-mono text-[0.6875rem] uppercase tracking-[0.1em]"
              style={{ color: `rgb(${rgb})` }}
            >
              channel · {sev}
            </span>
          </span>

          {/* Collapsed: a five-bar mini profile, so the pill still reports state
              without being opened. */}
          {!open && (
            <span className="ml-2 flex h-5 items-end gap-[3px]">
              {[
                (Math.log10(d.cn2) + 17) / 4,
                Math.min(1, d.jitterHz / 60),
                Math.min(1, d.jitterUrad / 2000),
                Math.min(1, d.awgnSigma / 20),
                d.occlusion,
              ].map((v, i) => (
                <span
                  key={i}
                  className="w-[3px] rounded-pill transition-[height] duration-300 ease-liquid"
                  style={{
                    height: `${Math.max(0.12, v) * 100}%`,
                    backgroundColor: `rgba(${rgb},${0.35 + 0.5 * v})`,
                  }}
                />
              ))}
            </span>
          )}

          <motion.span
            aria-hidden
            animate={{ rotate: open ? 180 : 0 }}
            transition={{ duration: 0.4, ease: [0.32, 0.72, 0, 1] }}
            className="ml-auto font-mono text-[0.625rem] text-readout-tertiary"
          >
            ▲
          </motion.span>
        </motion.button>

        {/* ------------------------------ body ------------------------------- */}
        <AnimatePresence initial={false}>
          {open && (
            <motion.div
              key="body"
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.42, ease: [0.32, 0.72, 0, 1] }}
              className="overflow-hidden border-t border-white/[0.07]"
            >
              <div className="max-h-[min(60vh,32rem)] overflow-y-auto scroll-frost px-4 pb-4 pt-1">
                {/* presets */}
                <div className="mb-1 mt-2 grid grid-cols-2 gap-2">
                  {PRESETS.map((p) => (
                    <button
                      key={p.id}
                      type="button"
                      onClick={() => setDisturbance(p.patch)}
                      className="tactile group rounded-lg px-2.5 py-2 text-left transition-colors duration-200 hover:border-photon/40 hover:bg-photon/[0.07]"
                    >
                      <span className="block font-mono text-[0.6875rem] uppercase tracking-[0.1em] text-readout-primary">
                        {p.label}
                      </span>
                      <span className="block text-[0.5625rem] leading-tight text-readout-dim">
                        {p.note}
                      </span>
                    </button>
                  ))}
                </div>

                <div className="my-3 h-px bg-white/[0.07]" />

                <TactileSlider
                  label="Refractive structure constant"
                  unit="m⁻²ᐟ³"
                  log
                  min={1e-17}
                  max={1e-13}
                  value={d.cn2}
                  onChange={(v) => setDisturbance({ cn2: v })}
                  format={() => `${cn2.mantissa}e${cn2.exponent}`}
                  detents={[0.25, 0.5, 0.75]}
                  cautionAt={0.62}
                  hint="Sets the Fried parameter r₀ and the Rytov variance. Raising it deepens
                        intensity fading, which starves the detector — that, not beam wander,
                        is how turbulence breaks coarse acquisition."
                />

                <TactileSlider
                  label="Platform jitter frequency"
                  unit="Hz"
                  min={0}
                  max={60}
                  step={0.5}
                  value={d.jitterHz}
                  onChange={(v) => setDisturbance({ jitterHz: v })}
                  format={(v) => v.toFixed(1)}
                  detents={[0.2, 0.5]}
                  hint="Centre frequency of the narrowband vibration resonance. Above roughly a
                        third of the loop rate the PID cannot track it, and the SAC tuner should
                        respond by trading proportional gain for damping."
                />

                <TactileSlider
                  label="Platform jitter amplitude"
                  unit="µrad RMS"
                  min={0}
                  max={2500}
                  step={10}
                  value={d.jitterUrad}
                  onChange={(v) => setDisturbance({ jitterUrad: v })}
                  format={(v) => v.toFixed(0)}
                  hint="Mechanical disturbance at the gimbal base. At this focal length 220 µrad
                        is about one pixel, so a milliradian of vibration is several pixels of
                        line-of-sight motion per frame."
                />

                <TactileSlider
                  label="Sensor noise"
                  unit="σ DN"
                  min={0}
                  max={24}
                  step={0.5}
                  value={d.awgnSigma}
                  onChange={(v) => setDisturbance({ awgnSigma: v })}
                  format={(v) => v.toFixed(1)}
                  hint="Additive white Gaussian noise in 8-bit counts. Degrades sub-pixel centroid
                        precision first and detection recall second."
                />

                <TactileSlider
                  label="Cloud occlusion"
                  unit="opacity"
                  min={0}
                  max={1}
                  step={0.01}
                  value={d.occlusion}
                  onChange={(v) => setDisturbance({ occlusion: v })}
                  format={(v) => `${(v * 100).toFixed(0)}%`}
                  cautionAt={0.5}
                  hint="Obscuration between terminals. Push past ~70% to force detection loss and
                        watch the ghost reticle take over pointing."
                />

                <div className="mt-3 flex items-center justify-between border-t border-white/[0.07] pt-3">
                  <span className="font-mono text-[0.5625rem] uppercase tracking-[0.14em] text-readout-dim">
                    severity {(severity * 100).toFixed(0)}%
                  </span>
                  <button
                    type="button"
                    onClick={resetDisturbance}
                    className="rounded-pill border border-white/12 bg-white/[0.05] px-3 py-1 font-mono text-[0.5625rem] uppercase tracking-[0.14em] text-readout-secondary transition-colors duration-200 hover:border-photon/40 hover:text-photon-200"
                  >
                    restore nominal
                  </button>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  );
}
