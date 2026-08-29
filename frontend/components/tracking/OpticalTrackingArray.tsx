'use client';

/**
 * SPECTRA GUARD — Optical Tracking Array.
 *
 * The centre column: the live virtual camera above, the trajectory and residual
 * plot below. They are separate glass sheets rather than one panel with a divider
 * because they answer different questions — "what does the camera see right now"
 * and "how has the filter behaved over the last fifteen seconds" — and giving
 * each its own edge lets the eye switch between them cleanly.
 *
 * The viewport panel is `flush`: its child owns the full bleed, so no padding
 * gets between the sensor image and the bezel.
 */

import { motion } from 'framer-motion';
import { GlassPanel, StatusChip } from '@/components/glass/GlassPanel';
import { LiveViewport } from './LiveViewport';
import { TrajectoryPlot } from './TrajectoryPlot';
import { useSlowTelemetry } from '@/lib/use-pat-link';
import { useReadout } from '@/lib/use-readout';
import type { TelemetryFrame } from '@/lib/types';

/** Compact header readout — one line of monospace, updated without re-rendering. */
function HeaderReadout({
  compute,
  tint = 'text-readout-secondary',
}: {
  compute: (f: TelemetryFrame | null) => string;
  tint?: string;
}) {
  const ref = useReadout(compute);
  return (
    <span
      ref={ref as React.RefObject<HTMLSpanElement>}
      className={`font-mono text-[0.625rem] uppercase tracking-[0.1em] tabular-nums ${tint}`}
    >
      —
    </span>
  );
}

export function OpticalTrackingArray({ className = '' }: { className?: string }) {
  const { stage } = useSlowTelemetry();
  const coasting = stage === 'COAST';

  return (
    <div className={`flex min-h-0 flex-col gap-3 ${className}`}>
      {/* ---------------------------- live feed ---------------------------- */}
      <GlassPanel
        flush
        blur="deep"
        accent={coasting ? 'ember' : stage === 'TRACK' ? 'photon' : 'none'}
        className="relative z-scene flex min-h-0 flex-1 flex-col"
        bodyClassName="min-h-0 flex-1"
        label="Optical tracking array · virtual camera"
        headerSlot={
          <div className="flex items-center gap-3">
            <HeaderReadout
              compute={(f) => (f ? `seq ${f.seq.toString().padStart(6, '0')}` : 'seq ——————')}
              tint="text-readout-tertiary"
            />
            <HeaderReadout
              compute={(f) => (f ? `${f.metrics.fps.toFixed(0)} fps` : '— fps')}
            />
            <StatusChip tone={coasting ? 'ember' : stage === 'TRACK' ? 'photon' : 'neutral'}>
              {coasting ? 'occluded' : stage === 'SEARCH' ? 'scanning' : 'nominal'}
            </StatusChip>
          </div>
        }
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
      >
        <LiveViewport className="absolute inset-0" />

        {/* Coast banner. Deliberately blunt: when the system is flying blind the
            operator must not have to infer it from a colour change. */}
        <motion.div
          className="pointer-events-none absolute inset-x-0 bottom-0 z-overlay flex justify-center pb-3"
          initial={false}
          animate={{ opacity: coasting ? 1 : 0, y: coasting ? 0 : 8 }}
          transition={{ duration: 0.32, ease: [0.16, 1, 0.3, 1] }}
        >
          <span className="glass-deep flex items-center gap-2 rounded-pill border border-ember/40 px-3 py-1.5">
            <span className="h-1.5 w-1.5 animate-breathe rounded-full bg-ember shadow-[0_0_8px_rgba(255,169,43,0.9)]" />
            <span className="font-mono text-[0.625rem] uppercase tracking-[0.14em] text-ember-400">
              beacon occluded — pointing from ekf prediction
            </span>
            <HeaderReadout
              compute={(f) => (f ? `${f.estimate.coastFrames}f` : '—')}
              tint="text-ember-400"
            />
          </span>
        </motion.div>
      </GlassPanel>

      {/* --------------------------- trajectory ---------------------------- */}
      <GlassPanel
        flush
        blur="frost"
        className="z-plot shrink-0"
        bodyClassName="h-[13.5rem]"
        label="Trajectory · actual vs estimated"
        headerSlot={
          <div className="flex items-center gap-3">
            <HeaderReadout
              compute={(f) => (f ? `rms ${f.metrics.rmsErrorPx.toFixed(2)} px` : 'rms — px')}
            />
            <HeaderReadout
              compute={(f) => (f ? `retention ${(f.metrics.lockRetention * 100).toFixed(1)}%` : '—')}
              tint="text-photon-200"
            />
          </div>
        }
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1], delay: 0.06 }}
      >
        <TrajectoryPlot />
      </GlassPanel>
    </div>
  );
}
