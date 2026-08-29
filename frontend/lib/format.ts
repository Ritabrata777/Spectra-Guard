/**
 * Formatting helpers for instrument readouts.
 *
 * Two rules drive everything here, both borrowed from real telemetry consoles:
 *
 * 1. Fixed decimal places, always. A value that renders "4.2" then "4.25" then
 *    "4" changes width and makes the whole column twitch. Pad, never trim.
 * 2. Sign is always explicit on error/rate channels. An operator reading a
 *    pointing error needs the direction as much as the magnitude, and a leading
 *    "+" also keeps positive and negative values the same width.
 */

export function fixed(v: number | null | undefined, dp = 2): string {
  if (v == null || !Number.isFinite(v)) return '—';
  return v.toFixed(dp);
}

/** Signed, fixed-width. Use for errors, rates, offsets. */
export function signed(v: number | null | undefined, dp = 2): string {
  if (v == null || !Number.isFinite(v)) return '—';
  return `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(dp)}`;
}

/** Degrees with a true degree sign. */
export function deg(v: number | null | undefined, dp = 3): string {
  if (v == null || !Number.isFinite(v)) return '—';
  return `${signed(v, dp)}°`;
}

export function pct(v: number | null | undefined, dp = 1): string {
  if (v == null || !Number.isFinite(v)) return '—';
  return `${(v * 100).toFixed(dp)}%`;
}

export function ms(v: number | null | undefined, dp = 0): string {
  if (v == null || !Number.isFinite(v)) return '—';
  return `${v.toFixed(dp)} ms`;
}

/**
 * Scientific notation for Cn², which spans 1e-17 (astronomical seeing) to
 * 1e-13 (severe near-ground turbulence). A linear readout would be useless
 * across four orders of magnitude, so render mantissa and exponent separately
 * and let the component style the exponent as a superscript.
 */
export function sci(v: number): { mantissa: string; exponent: number } {
  if (!Number.isFinite(v) || v === 0) return { mantissa: '0.00', exponent: 0 };
  const exponent = Math.floor(Math.log10(Math.abs(v)));
  const mantissa = (v / 10 ** exponent).toFixed(2);
  return { mantissa, exponent };
}

/** Clock-style elapsed time, mm:ss.t — how mission time is actually read. */
export function elapsed(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return '--:--.-';
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${String(m).padStart(2, '0')}:${s.toFixed(1).padStart(4, '0')}`;
}

/**
 * Map a normalised 0..1 quality figure onto the three-accent state palette.
 * Centralised so "what counts as degraded" is defined once, not re-guessed by
 * each widget with slightly different thresholds.
 */
export type Tone = 'photon' | 'ember' | 'plasma';

export function toneForConfidence(c: number): Tone {
  if (c >= 0.75) return 'photon';
  if (c >= 0.4) return 'ember';
  return 'plasma';
}

export function toneForError(errPx: number, tolerancePx = 12): Tone {
  if (errPx <= tolerancePx) return 'photon';
  if (errPx <= tolerancePx * 4) return 'ember';
  return 'plasma';
}

export const TONE_RGB: Record<Tone, [number, number, number]> = {
  photon: [34, 224, 255],
  ember: [255, 169, 43],
  plasma: [255, 59, 92],
};

export function rgba(tone: Tone, alpha: number): string {
  const [r, g, b] = TONE_RGB[tone];
  return `rgba(${r},${g},${b},${alpha})`;
}

/**
 * Exponential smoothing for display-only values.
 *
 * Applied to things like the FPS counter and the confidence ring, never to
 * anything logged. A raw 60 Hz FPS number is unreadable noise; a smoothed one
 * is informative. But smoothing a value that gets exported as evidence would be
 * quietly falsifying the record, so `metrics` on the wire stays raw and only
 * the pixels get filtered.
 */
export function ema(prev: number, next: number, alpha = 0.12): number {
  if (!Number.isFinite(prev)) return next;
  return prev + alpha * (next - prev);
}
