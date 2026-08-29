'use client';

/**
 * SPECTRA GUARD — Intelligence Readout sidebar.
 *
 * The numbers ISRO asked for, in the order an operator needs them: what the
 * system believes (confidence, stage), where it is pointing (pan/tilt), how
 * wrong it is (pixel and angular error), and whether it is keeping up (retention,
 * FPS, filter consistency).
 *
 * Only the PAT stage lives in React state — it changes a handful of times per
 * run. Every numeric value is bound through `useReadout` and updated by direct
 * DOM writes, so this panel costs one render per stage transition rather than
 * sixty renders per second.
 */

import { AnimatePresence, motion } from 'framer-motion';
import { GlassPanel, StatusChip } from '@/components/glass/GlassPanel';
import { ConfidenceRing } from './ConfidenceRing';
import { AxisPair, MagnitudeBar, TelemetryStat } from './TelemetryStat';
import { useSlowTelemetry } from '@/lib/use-pat-link';
import { useReadout } from '@/lib/use-readout';
import { elapsed, fixed, pct, sci, signed, TONE_RGB, toneForError } from '@/lib/format';
import { LOCK_TOLERANCE_PX, type PatStage, type TelemetryFrame } from '@/lib/types';

/** Stages the operator actually watches. IDLE and FAULT are shown as chips. */
const PIPELINE: PatStage[] = ['SEARCH', 'ACQUIRE', 'TRACK', 'COAST'];

const STAGE_TONE: Record<PatStage, 'neutral' | 'photon' | 'ember' | 'plasma'> = {
  IDLE: 'neutral',
  SEARCH: 'neutral',
  ACQUIRE: 'ember',
  TRACK: 'photon',
  COAST: 'ember',
  FAULT: 'plasma',
};

const STAGE_COPY: Record<PatStage, string> = {
  IDLE: 'Engine loaded. Arm a run to begin.',
  SEARCH: 'Spiral scan across the uncertainty cone.',
  ACQUIRE: 'Beacon seen. Filter converging on the track.',
  TRACK: 'Closed loop. Ready to hand off to fine pointing.',
  COAST: 'Beacon occluded. Driving servos from the EKF prediction.',
  FAULT: 'Coast timeout exceeded. Restarting the search.',
};

/**
 * The PAT pipeline as a stepper.
 *
 * This is the one place numbered/ordered markers are justified: the stages are a
 * real state machine with a real order, and showing the operator where they are
 * in it carries information. Using the same device for a list of unrelated
 * panels would be decoration.
 */
function StageStepper({ stage }: { stage: PatStage }) {
  const activeIdx = PIPELINE.indexOf(stage);
  return (
    <div className="flex items-center gap-1">
      {PIPELINE.map((s, i) => {
        const reached = activeIdx >= i && activeIdx !== -1;
        const isActive = stage === s;
        return (
          <div key={s} className="flex flex-1 flex-col gap-1.5">
            <div className="relative h-0.5 overflow-hidden rounded-pill bg-white/[0.08]">
              <motion.div
                className="absolute inset-y-0 left-0 rounded-pill"
                initial={false}
                animate={{
                  width: reached ? '100%' : '0%',
                  backgroundColor: isActive
                    ? stage === 'COAST'
                      ? 'rgb(255,169,43)'
                      : 'rgb(34,224,255)'
                    : 'rgba(34,224,255,0.35)',
                }}
                transition={{ duration: 0.45, ease: [0.32, 0.72, 0, 1] }}
              />
            </div>
            <span
              className={[
                'font-mono text-[0.5rem] uppercase tracking-[0.12em] transition-colors duration-300',
                isActive
                  ? stage === 'COAST'
                    ? 'text-ember'
                    : 'text-photon'
                  : reached
                    ? 'text-readout-secondary'
                    : 'text-readout-dim',
              ].join(' ')}
            >
              {s}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="mb-1 mt-4 flex items-center gap-2 first:mt-0">
      <span className="eyebrow whitespace-nowrap">{children}</span>
      <span className="h-px flex-1 bg-white/[0.07]" />
    </div>
  );
}

export function IntelligenceReadout({ className = '' }: { className?: string }) {
  const { stage, linkMode, detector } = useSlowTelemetry();
  const tone = STAGE_TONE[stage];

  return (
    <GlassPanel
      blur="abyssal"
      accent={tone === 'neutral' ? 'none' : tone}
      caustic
      className={`flex h-full flex-col ${className}`}
      bodyClassName="flex-1 overflow-y-auto scroll-frost"
      label="Intelligence readout"
      headerSlot={
        <StatusChip tone={tone} pulse={stage === 'TRACK' || stage === 'COAST'}>
          {stage}
        </StatusChip>
      }
      initial={{ opacity: 0, x: 24 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1], delay: 0.1 }}
    >
      {/* ---- confidence ---- */}
      <div className="flex flex-col items-center">
        <ConfidenceRing size={152} />
        <AnimatePresence mode="wait">
          <motion.p
            key={stage}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.28, ease: [0.16, 1, 0.3, 1] }}
            className="mt-1 max-w-[15rem] text-balance text-center text-[0.6875rem] leading-relaxed text-readout-secondary"
          >
            {STAGE_COPY[stage]}
          </motion.p>
        </AnimatePresence>
      </div>

      <div className="mt-4">
        <StageStepper stage={stage} />
      </div>

      {/* ---- pointing ---- */}
      <SectionLabel>Gimbal</SectionLabel>
      <AxisPair
        label="Commanded attitude"
        tagA="pan · az"
        tagB="tilt · el"
        unit="°"
        computeA={(f) => (f ? f.gimbal.pan.toFixed(3) : '—')}
        computeB={(f) => (f ? f.gimbal.tilt.toFixed(3) : '—')}
      />
      <AxisPair
        label="Slew rate"
        tagA="pan rate"
        tagB="tilt rate"
        unit="°/s"
        computeA={(f) => (f ? signed(f.gimbal.panRate, 2) : '—')}
        computeB={(f) => (f ? signed(f.gimbal.tiltRate, 2) : '—')}
      />

      {/* ---- error ---- */}
      <SectionLabel>Tracking error</SectionLabel>
      <MagnitudeBar
        label="Instantaneous miss"
        compute={(f) => (f ? `${f.error.norm.toFixed(2)} px` : '—')}
        // Normalised against four times the lock tolerance so the bar has
        // useful resolution in the region that matters instead of pinning.
        normalise={(f) => (f ? f.error.norm / (LOCK_TOLERANCE_PX * 4) : 0)}
        toneAt={(f) => TONE_RGB[toneForError(f ? f.error.norm : 0)]}
        markerAt={0.25}
        markerLabel={`lock tolerance ${LOCK_TOLERANCE_PX} px`}
      />
      <TelemetryStat
        label="Angular · az"
        unit="mdeg"
        compute={(f) => (f ? signed(f.error.azErr * 1000, 1) : '—')}
      />
      <TelemetryStat
        label="Angular · el"
        unit="mdeg"
        compute={(f) => (f ? signed(f.error.elErr * 1000, 1) : '—')}
      />
      <TelemetryStat
        label="RMS since lock"
        unit="px"
        compute={(f) => (f ? fixed(f.metrics.rmsErrorPx, 2) : '—')}
      />

      {/* ---- performance: the deliverable log, live ---- */}
      <SectionLabel>Performance</SectionLabel>
      <TelemetryStat
        label="Lock retention"
        size="lg"
        compute={(f) => (f ? pct(f.metrics.lockRetention, 1) : '—')}
      />
      <TelemetryStat
        label="Acquisition time"
        compute={(f) =>
          f && f.metrics.acqTimeMs != null ? `${f.metrics.acqTimeMs.toFixed(0)} ms` : 'pending'
        }
      />
      <TelemetryStat label="Loop rate" unit="fps" compute={(f) => (f ? fixed(f.metrics.fps, 1) : '—')} />
      <TelemetryStat
        label="Compute budget"
        unit="ms"
        compute={(f) => (f ? fixed(f.metrics.computeMs, 2) : '—')}
      />
      <TelemetryStat label="Mission time" compute={(f) => elapsed(f?.t)} />

      {/* ---- filter health ---- */}
      <SectionLabel>Filter health</SectionLabel>
      <TelemetryStat
        label="NIS · χ² 2dof"
        compute={(f) => (f ? fixed(f.metrics.nis, 2) : '—')}
      />
      <p className="mt-0.5 text-[0.625rem] leading-relaxed text-readout-dim">
        A consistent filter averages 2.0. Sustained values above the 9.21 gate mean
        the motion model no longer matches the target.
      </p>
      <TelemetryStat
        label="Blind frames"
        compute={(f) => (f ? String(f.estimate.coastFrames) : '—')}
      />
      <TelemetryStat
        label="Posterior 3σ"
        unit="px"
        compute={(f) => (f ? fixed(Math.max(f.estimate.sigmaX, f.estimate.sigmaY) * 3, 1) : '—')}
      />

      {/* ---- controller ---- */}
      <SectionLabel>Controller</SectionLabel>
      <div className="grid grid-cols-3 gap-2">
        {(['kp', 'ki', 'kd'] as const).map((k) => (
          <div key={k} className="tactile rounded-lg px-2 py-1.5 text-center">
            <div className="font-mono text-[0.5625rem] uppercase tracking-[0.14em] text-readout-dim">
              {k}
            </div>
            <TelemetryStatInline compute={(f) => (f ? f.control[k].toFixed(3) : '—')} />
          </div>
        ))}
      </div>
      <div className="mt-2 flex items-center justify-between">
        <span className="eyebrow">Tuner</span>
        <TelemetryChip compute={(f) => (f ? f.control.tunerMode : '—')} />
      </div>

      {/* ---- atmosphere ---- */}
      <SectionLabel>Channel</SectionLabel>
      <TelemetryStat
        label="Cn² structure const"
        unit="m⁻²ᐟ³"
        compute={(f) => {
          if (!f) return '—';
          const { mantissa, exponent } = sci(f.disturbance.cn2);
          return `${mantissa}e${exponent}`;
        }}
      />
      <MagnitudeBar
        label="Received irradiance"
        compute={(f) => (f ? `${(f.scintillation * 100).toFixed(0)}%` : '—')}
        normalise={(f) => (f ? f.scintillation : 0)}
        toneAt={(f) => (f && f.scintillation < 0.35 ? TONE_RGB.ember : TONE_RGB.photon)}
      />
      <TelemetryStat
        label="Platform jitter"
        unit="µrad"
        compute={(f) => (f ? f.disturbance.jitterUrad.toFixed(0) : '—')}
      />
      <TelemetryStat
        label="Slant range"
        unit="km"
        compute={(f) => (f ? fixed(f.truth.rangeKm, 2) : '—')}
      />

      {/* ---- provenance ---- */}
      <div className="mt-5 flex items-center justify-between border-t border-white/[0.07] pt-3">
        <span className="font-mono text-[0.5625rem] uppercase tracking-[0.14em] text-readout-dim">
          {detector === 'YOLOV11' ? 'yolov11 detector' : 'classical detector'}
        </span>
        <span
          className={[
            'font-mono text-[0.5625rem] uppercase tracking-[0.14em]',
            linkMode === 'LINK' ? 'text-photon-200' : 'text-ember-400',
          ].join(' ')}
        >
          {linkMode === 'LINK' ? 'engine link' : linkMode === 'SIM' ? 'browser sim' : 'offline'}
        </span>
      </div>
    </GlassPanel>
  );
}

/* --------------------------- small bound helpers -------------------------- */

function TelemetryStatInline({ compute }: { compute: (f: TelemetryFrame | null) => string }) {
  const ref = useReadout(compute);
  return (
    <span
      ref={ref as React.RefObject<HTMLSpanElement>}
      className="font-mono text-[0.8125rem] font-medium tabular-nums text-readout-primary"
    >
      —
    </span>
  );
}

function TelemetryChip({ compute }: { compute: (f: TelemetryFrame | null) => string }) {
  const ref = useReadout(compute);
  return (
    <span className="inline-flex items-center gap-1.5 rounded-pill border border-photon/35 bg-photon/[0.10] px-2 py-0.5">
      <span className="h-1 w-1 rounded-full bg-photon shadow-[0_0_6px_rgba(34,224,255,0.9)]" />
      <span
        ref={ref as React.RefObject<HTMLSpanElement>}
        className="font-mono text-eyebrow uppercase text-photon-200"
      >
        —
      </span>
    </span>
  );
}
