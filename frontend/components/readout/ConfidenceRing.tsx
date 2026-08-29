'use client';

/**
 * SPECTRA GUARD — Target Lock Confidence ring.
 *
 * A canvas ring, repainted at 60 Hz from `bus.latest`, showing the fused lock
 * confidence. Canvas rather than an animated SVG `stroke-dashoffset` for a
 * concrete reason: animating an SVG attribute at 60 Hz forces the browser to
 * re-run layout and re-rasterise the path every frame, and with a blur filter
 * behind it that is genuinely expensive. A canvas arc is a single GPU-composited
 * draw call and the glass panels behind it never get invalidated.
 *
 * The value is critically damped toward its target rather than snapped. Real
 * needle instruments have inertia, and a confidence figure that snaps between
 * 0.4 and 0.9 on consecutive frames is unreadable. Damping is presentation only
 * — the logged value stays raw.
 */

import { useEffect, useRef } from 'react';
import { bus } from '@/lib/telemetry-bus';
import { TONE_RGB, toneForConfidence } from '@/lib/format';

export function ConfidenceRing({ size = 148 }: { size?: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const valueRef = useRef<HTMLSpanElement>(null);
  const labelRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    const cx = size / 2;
    const cy = size / 2;
    const r = size / 2 - 14;
    const START = -Math.PI * 0.75; // open at the bottom, gauge-style
    const SWEEP = Math.PI * 1.5;

    // Critically damped second-order follower: position + velocity, no overshoot.
    let shown = 0;
    let vel = 0;
    let raf = 0;

    const draw = (now: number) => {
      raf = requestAnimationFrame(draw);
      const f = bus.latest;
      const target = f ? f.confidence : 0;

      // Spring-damper. omega sets the response speed; zeta = 1 is critical
      // damping, which settles fastest without oscillating past the value —
      // overshoot on a confidence gauge would imply certainty we do not have.
      const dt = 1 / 60;
      const omega = 11;
      const accel = omega * omega * (target - shown) - 2 * omega * vel;
      vel += accel * dt;
      shown += vel * dt;
      shown = Math.max(0, Math.min(1, shown));

      const [tr, tg, tb] = TONE_RGB[toneForConfidence(shown)];

      ctx.clearRect(0, 0, size, size);

      /* ---- subsurface caustic: light behind the glass, not on it ---- */
      const bleed = ctx.createRadialGradient(cx, cy, r * 0.35, cx, cy, r * 1.25);
      bleed.addColorStop(0, `rgba(${tr},${tg},${tb},${0.20 * shown})`);
      bleed.addColorStop(1, `rgba(${tr},${tg},${tb},0)`);
      ctx.fillStyle = bleed;
      ctx.fillRect(0, 0, size, size);

      /* ---- track ---- */
      ctx.lineCap = 'round';
      ctx.lineWidth = 7;
      ctx.strokeStyle = 'rgba(255,255,255,0.07)';
      ctx.beginPath();
      ctx.arc(cx, cy, r, START, START + SWEEP);
      ctx.stroke();

      /* ---- graduations every 10%: makes it an instrument, not a progress bar ---- */
      for (let i = 0; i <= 10; i++) {
        const a = START + (SWEEP * i) / 10;
        const major = i % 5 === 0;
        const inner = r + (major ? 7 : 9);
        const outer = r + 12;
        ctx.strokeStyle = major ? 'rgba(232,238,247,0.34)' : 'rgba(232,238,247,0.16)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(cx + Math.cos(a) * inner, cy + Math.sin(a) * inner);
        ctx.lineTo(cx + Math.cos(a) * outer, cy + Math.sin(a) * outer);
        ctx.stroke();
      }

      /* ---- value arc, with the glow drawn as a real shadow ---- */
      if (shown > 0.002) {
        ctx.save();
        ctx.shadowColor = `rgba(${tr},${tg},${tb},0.85)`;
        ctx.shadowBlur = 16;
        const grad = ctx.createLinearGradient(0, cy - r, 0, cy + r);
        grad.addColorStop(0, `rgba(${tr},${tg},${tb},1)`);
        grad.addColorStop(1, `rgba(${tr},${tg},${tb},0.55)`);
        ctx.strokeStyle = grad;
        ctx.lineWidth = 7;
        ctx.beginPath();
        ctx.arc(cx, cy, r, START, START + SWEEP * shown);
        ctx.stroke();
        ctx.restore();

        // Leading cap: a bright dot at the head of the arc reads as the live
        // edge of the value, the way a needle tip does.
        const a = START + SWEEP * shown;
        const hx = cx + Math.cos(a) * r;
        const hy = cy + Math.sin(a) * r;
        ctx.save();
        ctx.shadowColor = `rgba(${tr},${tg},${tb},1)`;
        ctx.shadowBlur = 14;
        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(hx, hy, 3, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }

      /* ---- coast indicator: a second, dimmer arc showing the raw (undamped)
             value, so the operator can see the instantaneous figure the log
             records alongside the smoothed needle ---- */
      if (f && Math.abs(target - shown) > 0.04) {
        const a = START + SWEEP * target;
        ctx.strokeStyle = `rgba(${tr},${tg},${tb},0.35)`;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(cx, cy, r + 5, a - 0.02, a + 0.02);
        ctx.stroke();
      }

      /* ---- pulse ring while genuinely locked ---- */
      if (!reduceMotion && shown > 0.75) {
        const p = (now / 1600) % 1;
        ctx.strokeStyle = `rgba(${tr},${tg},${tb},${0.22 * (1 - p)})`;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(cx, cy, r * (0.72 + p * 0.34), 0, Math.PI * 2);
        ctx.stroke();
      }

      /* ---- centre text, written directly to the DOM (no React render) ---- */
      const pctStr = `${(shown * 100).toFixed(1)}`;
      if (valueRef.current && valueRef.current.textContent !== pctStr) {
        valueRef.current.textContent = pctStr;
        valueRef.current.style.color = `rgb(${tr},${tg},${tb})`;
        valueRef.current.style.textShadow = `0 0 18px rgba(${tr},${tg},${tb},0.55)`;
      }
      const stageStr = f ? f.stage : 'IDLE';
      if (labelRef.current && labelRef.current.textContent !== stageStr) {
        labelRef.current.textContent = stageStr;
      }
    };

    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [size]);

  return (
    <div className="relative flex items-center justify-center" style={{ width: size, height: size }}>
      <canvas ref={canvasRef} className="block" />
      <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
        <div className="flex items-baseline gap-0.5">
          <span
            ref={valueRef}
            className="font-mono text-numeral font-medium tabular-nums text-photon"
          >
            0.0
          </span>
          <span className="font-mono text-micro text-readout-tertiary">%</span>
        </div>
        <span
          ref={labelRef}
          className="mt-0.5 font-mono text-eyebrow uppercase text-readout-secondary"
        >
          IDLE
        </span>
      </div>
    </div>
  );
}
