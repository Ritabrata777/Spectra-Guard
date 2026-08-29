'use client';

/**
 * Instrument readout primitives.
 *
 * Every value here is bound through `useReadout`, so these components render
 * once and are then updated by direct DOM writes at 60 Hz. See
 * `lib/use-readout.ts` for why.
 */

import { useReadout, useStyleReadout } from '@/lib/use-readout';
import type { TelemetryFrame } from '@/lib/types';

/**
 * A single labelled channel.
 *
 * The label is a short channel tag (PAN·AZ, ERR·PX) rather than a sentence.
 * That is the vernacular of real telemetry: an operator scanning a column of
 * twelve values reads tags, not prose, and fixed-width tags let the eye travel
 * straight down the value column without re-anchoring.
 */
export function TelemetryStat({
  label,
  compute,
  unit,
  size = 'md',
  align = 'right',
}: {
  label: string;
  compute: (f: TelemetryFrame | null) => string;
  unit?: string;
  size?: 'sm' | 'md' | 'lg';
  align?: 'left' | 'right';
}) {
  const ref = useReadout(compute);

  const valueClass = {
    sm: 'text-readout',
    md: 'text-[0.9375rem]',
    lg: 'text-numeral',
  }[size];

  return (
    <div className="flex items-baseline justify-between gap-3 py-1.5">
      <span className="eyebrow shrink-0">{label}</span>
      <span
        className={[
          'flex items-baseline gap-1',
          align === 'right' ? 'justify-end' : 'justify-start',
        ].join(' ')}
      >
        <span
          ref={ref as React.RefObject<HTMLSpanElement>}
          className={`font-mono ${valueClass} font-medium tabular-nums text-readout-primary text-crisp`}
        >
          —
        </span>
        {unit && <span className="font-mono text-micro text-readout-tertiary">{unit}</span>}
      </span>
    </div>
  );
}

/**
 * Paired axis readout — pan and tilt belong together and reading them as one
 * two-column unit is faster than as two separate rows.
 */
export function AxisPair({
  label,
  computeA,
  computeB,
  tagA,
  tagB,
  unit,
}: {
  label: string;
  computeA: (f: TelemetryFrame | null) => string;
  computeB: (f: TelemetryFrame | null) => string;
  tagA: string;
  tagB: string;
  unit?: string;
}) {
  const refA = useReadout(computeA);
  const refB = useReadout(computeB);

  return (
    <div className="py-1.5">
      <span className="eyebrow">{label}</span>
      <div className="mt-1.5 grid grid-cols-2 gap-2">
        {[
          { tag: tagA, ref: refA },
          { tag: tagB, ref: refB },
        ].map(({ tag, ref }) => (
          <div
            key={tag}
            className="tactile rounded-lg px-2.5 py-1.5"
          >
            <div className="font-mono text-[0.5625rem] uppercase tracking-[0.16em] text-readout-dim">
              {tag}
            </div>
            <div className="flex items-baseline gap-1">
              <span
                ref={ref as React.RefObject<HTMLSpanElement>}
                className="font-mono text-[0.9375rem] font-medium tabular-nums text-readout-primary"
              >
                —
              </span>
              {unit && (
                <span className="font-mono text-[0.625rem] text-readout-tertiary">{unit}</span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * Horizontal magnitude bar, style-bound at 60 Hz.
 *
 * Used for pixel error against the lock tolerance. The tolerance is drawn as a
 * fixed tick so the bar answers "are we in spec" at a glance, not just "how big
 * is the number" — which is the actual question during a demo.
 */
export function MagnitudeBar({
  label,
  compute,
  normalise,
  toneAt,
  markerAt,
  markerLabel,
}: {
  label: string;
  compute: (f: TelemetryFrame | null) => string;
  /** Map the raw value to 0..1 of bar width. */
  normalise: (f: TelemetryFrame | null) => number;
  /** Returns an rgb triple for the current value. */
  toneAt: (f: TelemetryFrame | null) => [number, number, number];
  /** Fractional position of the spec marker, 0..1. */
  markerAt?: number;
  markerLabel?: string;
}) {
  const textRef = useReadout(compute);
  const barRef = useStyleReadout((f) => {
    const v = Math.max(0, Math.min(1, normalise(f)));
    const [r, g, b] = toneAt(f);
    return {
      width: `${(v * 100).toFixed(2)}%`,
      backgroundColor: `rgba(${r},${g},${b},0.9)`,
      boxShadow: `0 0 12px rgba(${r},${g},${b},0.65)`,
    } as Partial<CSSStyleDeclaration>;
  });

  return (
    <div className="py-1.5">
      <div className="flex items-baseline justify-between gap-3">
        <span className="eyebrow">{label}</span>
        <span
          ref={textRef as React.RefObject<HTMLSpanElement>}
          className="font-mono text-[0.9375rem] font-medium tabular-nums text-readout-primary text-crisp"
        >
          —
        </span>
      </div>
      <div className="tactile relative mt-2 h-1.5 overflow-hidden rounded-pill">
        <div
          ref={barRef as React.RefObject<HTMLDivElement>}
          className="h-full rounded-pill transition-[background-color] duration-300"
          style={{ width: '0%' }}
        />
        {markerAt != null && (
          <div
            aria-hidden
            className="absolute top-0 h-full w-px bg-white/40"
            style={{ left: `${markerAt * 100}%` }}
            title={markerLabel}
          />
        )}
      </div>
      {markerLabel && (
        <div className="mt-1 text-right font-mono text-[0.5625rem] uppercase tracking-[0.14em] text-readout-dim">
          {markerLabel}
        </div>
      )}
    </div>
  );
}
