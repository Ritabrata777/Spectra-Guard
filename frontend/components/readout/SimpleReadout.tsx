'use client';

import { AnimatePresence, motion } from 'framer-motion';
import { GlassPanel, StatusChip } from '@/components/glass/GlassPanel';
import { ConfidenceRing } from './ConfidenceRing';
import { MagnitudeBar, TelemetryStat } from './TelemetryStat';
import { useSlowTelemetry } from '@/lib/use-pat-link';
import { fixed, pct, TONE_RGB, toneForError } from '@/lib/format';
import { LOCK_TOLERANCE_PX, type PatStage } from '@/lib/types';

const TONE: Record<PatStage, 'neutral' | 'photon' | 'ember' | 'plasma'> = {
  IDLE: 'neutral', SEARCH: 'neutral', ACQUIRE: 'ember', TRACK: 'photon', COAST: 'ember', FAULT: 'plasma',
};
const MESSAGE: Record<PatStage, string> = {
  IDLE: 'Choose a target, then start tracking.',
  SEARCH: 'Scanning for the beacon.',
  ACQUIRE: 'Beacon found. Establishing lock.',
  TRACK: 'Locked. Pointing is stable.',
  COAST: 'Signal lost. Holding course from prediction.',
  FAULT: 'Lock was lost. Restart tracking.',
};

/** The operator view: status, error and outcome—not every engineering variable. */
export function SimpleReadout({ className = '' }: { className?: string }) {
  const { stage, linkMode } = useSlowTelemetry();
  const tone = TONE[stage];
  return (
    <GlassPanel
      blur="abyssal"
      accent={tone === 'neutral' ? 'none' : tone}
      caustic
      className={`flex h-full flex-col ${className}`}
      bodyClassName="flex-1"
      label="Tracking status"
      headerSlot={<StatusChip tone={tone} pulse={stage === 'TRACK' || stage === 'COAST'}>{stage}</StatusChip>}
      initial={{ opacity: 0, x: 24 }} animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1], delay: 0.1 }}
    >
      <div className="flex flex-col items-center">
        <ConfidenceRing size={140} />
        <AnimatePresence mode="wait">
          <motion.p key={stage} initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -5 }}
            className="mt-2 max-w-[15rem] text-center text-sm leading-relaxed text-readout-secondary">
            {MESSAGE[stage]}
          </motion.p>
        </AnimatePresence>
      </div>

      <div className="mt-6 space-y-4">
        <MagnitudeBar
          label="Pointing error"
          compute={(f) => (f ? `${f.error.norm.toFixed(1)} px` : '—')}
          normalise={(f) => (f ? f.error.norm / (LOCK_TOLERANCE_PX * 4) : 0)}
          toneAt={(f) => TONE_RGB[toneForError(f ? f.error.norm : 0)]}
          markerAt={0.25}
          markerLabel={`target: under ${LOCK_TOLERANCE_PX} px`}
        />
        <div className="grid grid-cols-2 gap-3">
          <TelemetryStat label="Lock retention" size="lg" compute={(f) => (f ? pct(f.metrics.lockRetention, 0) : '—')} />
          <TelemetryStat label="Acquisition" compute={(f) => f?.metrics.acqTimeMs != null ? `${f.metrics.acqTimeMs.toFixed(0)} ms` : 'waiting'} />
          <TelemetryStat label="Signal" compute={(f) => (f ? `${(f.scintillation * 100).toFixed(0)}%` : '—')} />
          <TelemetryStat label="Average error" unit="px" compute={(f) => (f ? fixed(f.metrics.rmsErrorPx, 2) : '—')} />
        </div>
      </div>

      <div className="mt-auto flex items-center justify-between border-t border-white/[0.07] pt-4 font-mono text-[0.625rem] uppercase tracking-[0.12em] text-readout-tertiary">
        <span>{linkMode === 'LINK' ? 'live engine' : linkMode === 'SIM' ? 'demo simulation' : 'offline'}</span>
        <span>details in event log</span>
      </div>
    </GlassPanel>
  );
}
