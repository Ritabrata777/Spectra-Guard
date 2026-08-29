'use client';

/**
 * Tactile slider.
 *
 * Built on pointer events rather than `<input type="range">`. That is a
 * deliberate cost: the native control gives you accessibility and keyboard
 * handling for free, so we re-implement both explicitly below (role="slider",
 * full arrow/Home/End/PageUp handling, aria-valuetext). What it cannot give us
 * is a recessed track that reads as machined into glass, a fill that glows from
 * inside the groove, and a thumb that grows under the finger — and the physical
 * feel of these controls is a large part of why this console reads as an
 * instrument rather than a form.
 *
 * LOG SCALE. Cn² spans 1e-17 to 1e-13. On a linear slider the entire useful
 * range of atmospheric conditions is crushed into the last two percent of travel.
 * A log-mapped axis makes each decade an equal, grabbable distance, which matches
 * how the quantity actually behaves: what matters is the order of magnitude.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

export interface TactileSliderProps {
  label: string;
  /** Short unit shown next to the value. */
  unit?: string;
  value: number;
  min: number;
  max: number;
  /** Linear step. Ignored when `log` is set. */
  step?: number;
  /** Map travel logarithmically — for quantities that span decades. */
  log?: boolean;
  onChange: (v: number) => void;
  /** Render the numeric value. Defaults to a 2-dp fixed string. */
  format?: (v: number) => string;
  /** Fractional positions (0..1) to draw as detents, e.g. nominal values. */
  detents?: number[];
  /** Accent the fill amber/red once the value passes this fraction of travel. */
  cautionAt?: number;
  /** Short explanation shown under the track. Physics, not marketing. */
  hint?: string;
}

export function TactileSlider({
  label,
  unit,
  value,
  min,
  max,
  step,
  log = false,
  onChange,
  format,
  detents,
  cautionAt = 0.7,
  hint,
}: TactileSliderProps) {
  const trackRef = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState(false);

  /* -------------------- value <-> normalised travel -------------------- */

  const toNorm = useCallback(
    (v: number) => {
      if (log) {
        const lo = Math.log10(min);
        const hi = Math.log10(max);
        return (Math.log10(Math.max(min, Math.min(max, v))) - lo) / (hi - lo);
      }
      return (Math.max(min, Math.min(max, v)) - min) / (max - min);
    },
    [log, min, max],
  );

  const fromNorm = useCallback(
    (t: number) => {
      const c = Math.max(0, Math.min(1, t));
      if (log) {
        const lo = Math.log10(min);
        const hi = Math.log10(max);
        return 10 ** (lo + c * (hi - lo));
      }
      const raw = min + c * (max - min);
      if (!step) return raw;
      // Snap, then clamp: snapping first can push the value a hair outside the
      // range when (max-min) is not an exact multiple of step.
      return Math.max(min, Math.min(max, Math.round(raw / step) * step));
    },
    [log, min, max, step],
  );

  const norm = toNorm(value);
  const display = format ? format(value) : value.toFixed(2);

  /* ----------------------------- pointer ------------------------------- */

  const applyFromClientX = useCallback(
    (clientX: number) => {
      const el = trackRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      onChange(fromNorm((clientX - rect.left) / Math.max(1, rect.width)));
    },
    [fromNorm, onChange],
  );

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    // Pointer capture keeps the drag alive when the finger leaves the 6 px track —
    // without it, sliders in a dense panel are almost impossible to use.
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    setDragging(true);
    applyFromClientX(e.clientX);
  };

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!dragging) return;
    applyFromClientX(e.clientX);
  };

  const endDrag = (e: React.PointerEvent<HTMLDivElement>) => {
    if ((e.currentTarget as HTMLElement).hasPointerCapture(e.pointerId)) {
      (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId);
    }
    setDragging(false);
  };

  /* ---------------------------- keyboard ------------------------------- */

  const onKeyDown = (e: React.KeyboardEvent) => {
    // Step in normalised travel so log axes move by a consistent perceptual
    // amount per keypress instead of crawling at the bottom of the decade.
    const fine = 0.01;
    const coarse = 0.1;
    let next: number | null = null;
    switch (e.key) {
      case 'ArrowRight':
      case 'ArrowUp':
        next = norm + (e.shiftKey ? coarse : fine);
        break;
      case 'ArrowLeft':
      case 'ArrowDown':
        next = norm - (e.shiftKey ? coarse : fine);
        break;
      case 'PageUp':
        next = norm + coarse;
        break;
      case 'PageDown':
        next = norm - coarse;
        break;
      case 'Home':
        next = 0;
        break;
      case 'End':
        next = 1;
        break;
      default:
        return;
    }
    e.preventDefault();
    onChange(fromNorm(next));
  };

  /* Cursor feedback while dragging, applied to the document so it survives the
     pointer leaving the thumb. */
  useEffect(() => {
    if (!dragging) return;
    const prev = document.body.style.cursor;
    document.body.style.cursor = 'grabbing';
    document.body.style.userSelect = 'none';
    return () => {
      document.body.style.cursor = prev;
      document.body.style.userSelect = '';
    };
  }, [dragging]);

  const caution = norm >= cautionAt;
  const rgb = caution ? '255,169,43' : '34,224,255';

  return (
    <div className="select-none py-2">
      <div className="mb-1.5 flex items-baseline justify-between gap-3">
        <span className="eyebrow">{label}</span>
        <span className="flex items-baseline gap-1">
          <span
            className="font-mono text-[0.9375rem] font-medium tabular-nums transition-colors duration-300"
            style={{ color: caution ? 'rgb(255,192,98)' : '#E8EEF7' }}
          >
            {display}
          </span>
          {unit && <span className="font-mono text-[0.625rem] text-readout-tertiary">{unit}</span>}
        </span>
      </div>

      <div
        ref={trackRef}
        role="slider"
        tabIndex={0}
        aria-label={label}
        aria-valuemin={min}
        aria-valuemax={max}
        aria-valuenow={value}
        aria-valuetext={`${display}${unit ? ` ${unit}` : ''}`}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onKeyDown={onKeyDown}
        // Generous hit area around a thin visual track: the groove reads as
        // precise, but the target is finger-sized.
        className="tactile relative h-6 cursor-grab touch-none rounded-pill active:cursor-grabbing"
      >
        {/* the groove */}
        <div className="pointer-events-none absolute inset-x-2 top-1/2 h-1.5 -translate-y-1/2 overflow-hidden rounded-pill bg-black/45 shadow-inset-tactile">
          <div
            className="h-full rounded-pill transition-colors duration-300"
            style={{
              width: `${norm * 100}%`,
              background: `linear-gradient(90deg, rgba(${rgb},0.35), rgba(${rgb},0.95))`,
              boxShadow: `0 0 14px rgba(${rgb},0.65)`,
            }}
          />
        </div>

        {/* detents */}
        {detents?.map((d) => (
          <span
            key={d}
            aria-hidden
            className="pointer-events-none absolute top-1/2 h-2.5 w-px -translate-y-1/2 bg-white/25"
            style={{ left: `calc(0.5rem + ${d} * (100% - 1rem))` }}
          />
        ))}

        {/* thumb */}
        <div
          aria-hidden
          className="pointer-events-none absolute top-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full border border-white/40 bg-gradient-to-b from-white/95 to-white/70 transition-[width,height,box-shadow] duration-150 ease-liquid"
          style={{
            left: `calc(0.5rem + ${norm} * (100% - 1rem))`,
            width: dragging ? 16 : 12,
            height: dragging ? 16 : 12,
            boxShadow: `0 2px 8px rgba(0,0,0,0.7), 0 0 ${dragging ? 18 : 10}px rgba(${rgb},${dragging ? 0.9 : 0.55})`,
          }}
        />
      </div>

      {hint && (
        <p className="mt-1.5 text-[0.625rem] leading-relaxed text-readout-dim">{hint}</p>
      )}
    </div>
  );
}
