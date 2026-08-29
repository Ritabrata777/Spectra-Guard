'use client';

/**
 * SPECTRA GUARD — Live Optical Viewport.
 *
 * The centre-stage instrument: the virtual camera feed with the tracker's own
 * beliefs drawn over it.
 *
 * ARCHITECTURE — why this is one canvas and zero React state.
 *
 * Everything on screen here updates at 60 Hz: the image, the bounding box, the
 * crosshair, the reticle, the uncertainty ellipse. If any of it were React
 * state, we would be asking React to reconcile a tree 60 times a second and the
 * canvas would miss vsync. Instead a single requestAnimationFrame loop reads
 * `bus.latest` and paints. The component renders exactly once.
 *
 * THE SIGNATURE ELEMENT — the covariance ghost reticle.
 *
 * When the beacon is occluded, most trackers either freeze the box or hide it.
 * Both lie: the first implies the target is still seen, the second implies the
 * system knows nothing. Neither is true — the EKF is still propagating, and its
 * uncertainty is growing at a rate the physics dictates. So on detection loss
 * the solid box gives way to a dashed ghost reticle at the filter's predicted
 * position, wrapped in the 3-sigma error ellipse taken straight from the
 * eigen-decomposition of the posterior covariance. The ellipse visibly inflates
 * the longer the target stays hidden, and snaps back down the instant a
 * measurement is accepted. The jury watches the Kalman filter think.
 */

import { motion } from 'framer-motion';
import { useEffect, useRef } from 'react';
import { bus } from '@/lib/telemetry-bus';
import { rgba, toneForConfidence, type Tone } from '@/lib/format';
import { LOCK_TOLERANCE_PX } from '@/lib/types';

const IMG_W = 640;
const IMG_H = 480;

export function LiveViewport({ className = '' }: { className?: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;

    const ctx = canvas.getContext('2d', { alpha: true });
    if (!ctx) return;

    const reduceMotion =
      typeof window !== 'undefined' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    /* ----------------------------- sizing ------------------------------ */

    let cssW = 0;
    let cssH = 0;

    const resize = () => {
      const rect = wrap.getBoundingClientRect();
      // Cap DPR at 2: beyond that we are pushing four times the pixels for a
      // difference nobody can see on a projector, and the fill rate is what
      // limits us on integrated graphics.
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      cssW = rect.width;
      cssH = rect.height;
      canvas.width = Math.max(1, Math.round(cssW * dpr));
      canvas.height = Math.max(1, Math.round(cssH * dpr));
      canvas.style.width = `${cssW}px`;
      canvas.style.height = `${cssH}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(wrap);

    /* -------------------------- draw helpers --------------------------- */

    /**
     * Corner brackets rather than a closed rectangle.
     *
     * A full rect around a small beacon hides the thing you are trying to look
     * at. Brackets leave the target clear, and they are the convention in real
     * optical tracking displays for exactly that reason.
     */
    const brackets = (
      x: number,
      y: number,
      w: number,
      h: number,
      arm: number,
      lw: number,
    ) => {
      const x0 = x - w / 2;
      const y0 = y - h / 2;
      const x1 = x + w / 2;
      const y1 = y + h / 2;
      ctx.lineWidth = lw;
      ctx.lineCap = 'round';
      ctx.beginPath();
      // top-left
      ctx.moveTo(x0, y0 + arm);
      ctx.lineTo(x0, y0);
      ctx.lineTo(x0 + arm, y0);
      // top-right
      ctx.moveTo(x1 - arm, y0);
      ctx.lineTo(x1, y0);
      ctx.lineTo(x1, y0 + arm);
      // bottom-right
      ctx.moveTo(x1, y1 - arm);
      ctx.lineTo(x1, y1);
      ctx.lineTo(x1 - arm, y1);
      // bottom-left
      ctx.moveTo(x0 + arm, y1);
      ctx.lineTo(x0, y1);
      ctx.lineTo(x0, y1 - arm);
      ctx.stroke();
    };

    /* ------------------------------ loop ------------------------------- */

    let raf = 0;
    // Smoothed presentation values. The underlying telemetry is honest and
    // unfiltered; these exist only so the drawn box does not visibly buzz at
    // pixel scale, which would read as a rendering bug rather than as noise.
    let boxW = 24;
    let boxH = 24;
    let lockPulse = 0;
    let prevStage = '';

    const draw = (now: number) => {
      raf = requestAnimationFrame(draw);
      const f = bus.latest;

      ctx.clearRect(0, 0, cssW, cssH);
      if (cssW < 2 || cssH < 2) return;

      /* ---- letterbox the sensor image into the viewport ---- */
      const scale = Math.min(cssW / IMG_W, cssH / IMG_H);
      const dw = IMG_W * scale;
      const dh = IMG_H * scale;
      const ox = (cssW - dw) / 2;
      const oy = (cssH - dh) / 2;
      // Image space -> canvas space.
      const tx = (u: number) => ox + u * scale;
      const ty = (v: number) => oy + v * scale;

      /* ---- the camera feed ---- */
      const img = bus.image;
      if (img) {
        // Nearest-neighbour: this is sensor data, and smoothing it invents
        // detail that the detector never saw.
        ctx.imageSmoothingEnabled = false;
        try {
          ctx.drawImage(img as CanvasImageSource, ox, oy, dw, dh);
        } catch {
          /* bitmap closed mid-frame; next frame will have a fresh one */
        }
        ctx.imageSmoothingEnabled = true;
      } else {
        ctx.fillStyle = '#04060c';
        ctx.fillRect(ox, oy, dw, dh);
        ctx.fillStyle = 'rgba(148,166,188,0.5)';
        ctx.font = '500 12px "IBM Plex Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText('NO OPTICAL FEED — ARM A RUN TO BEGIN', ox + dw / 2, oy + dh / 2);
        ctx.textAlign = 'left';
      }

      if (!f) return;

      const tone: Tone = toneForConfidence(f.confidence);
      const coasting = f.stage === 'COAST';
      const searching = f.stage === 'SEARCH';

      if (f.stage !== prevStage) {
        lockPulse = f.stage === 'TRACK' ? 1 : 0;
        prevStage = f.stage;
      }
      lockPulse = Math.max(0, lockPulse - 0.02);

      /* ---- boresight crosshair ---- */
      // The camera's optical axis: where the beam would point right now. The gap
      // in the middle exists so the crosshair never covers the beacon it is
      // trying to centre.
      const bx = tx(IMG_W / 2);
      const by = ty(IMG_H / 2);
      const gap = 7;
      const arm = 16;
      ctx.strokeStyle = 'rgba(232,238,247,0.42)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(bx - gap - arm, by);
      ctx.lineTo(bx - gap, by);
      ctx.moveTo(bx + gap, by);
      ctx.lineTo(bx + gap + arm, by);
      ctx.moveTo(bx, by - gap - arm);
      ctx.lineTo(bx, by - gap);
      ctx.moveTo(bx, by + gap);
      ctx.lineTo(bx, by + gap + arm);
      ctx.stroke();

      // Lock tolerance annulus — the acceptance region for declaring TRACK.
      // Drawing it makes the lock criterion visible instead of a hidden
      // constant, which is the first thing an evaluator asks about.
      ctx.strokeStyle = rgba(tone, 0.22);
      ctx.setLineDash([3, 4]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(bx, by, LOCK_TOLERANCE_PX * scale, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);

      /* ---- search sweep ---- */
      if (searching && !reduceMotion) {
        const phase = (now / 1400) % 1;
        const sweepX = ox + phase * dw;
        const grad = ctx.createLinearGradient(sweepX - 60, 0, sweepX + 60, 0);
        grad.addColorStop(0, 'rgba(34,224,255,0)');
        grad.addColorStop(0.5, 'rgba(34,224,255,0.10)');
        grad.addColorStop(1, 'rgba(34,224,255,0)');
        ctx.fillStyle = grad;
        ctx.fillRect(Math.max(ox, sweepX - 60), oy, 120, dh);
      }

      /* ---- detection: solid glowing brackets ---- */
      if (f.detection) {
        const d = f.detection;
        // Ease the box size so a jittering detector width does not strobe.
        boxW += (Math.max(d.w, 18) - boxW) * 0.22;
        boxH += (Math.max(d.h, 18) - boxH) * 0.22;

        const cx = tx(d.cx);
        const cy = ty(d.cy);
        const w = boxW * scale * 2.1;
        const h = boxH * scale * 2.1;

        // The glow is a real shadowBlur rather than a second stroked rect: it
        // falls off correctly and reads as light rather than as an outline.
        ctx.save();
        ctx.shadowColor = rgba(tone, 0.9);
        ctx.shadowBlur = 14 + 10 * lockPulse;
        ctx.strokeStyle = rgba(tone, 0.95);
        const pulseScale = 1 + 0.35 * lockPulse * lockPulse;
        brackets(cx, cy, w * pulseScale, h * pulseScale, Math.max(5, w * 0.24), 1.6);
        ctx.restore();

        // Centroid tick — sub-pixel position, drawn small so it reads as a
        // measurement rather than as decoration.
        ctx.fillStyle = rgba(tone, 0.95);
        ctx.fillRect(cx - 1, cy - 1, 2, 2);

        // Confidence figure pinned to the box.
        ctx.font = '500 10px "IBM Plex Mono", monospace';
        ctx.fillStyle = rgba(tone, 0.85);
        ctx.fillText(`${(d.score * 100).toFixed(0)}%`, cx + w / 2 + 6, cy - h / 2 + 4);
      }

      /* ---- THE GHOST RETICLE: EKF prediction + 3σ covariance ellipse ---- */
      if (coasting || (!f.detection && f.estimate.coastFrames > 0)) {
        const ex = tx(f.estimate.cx);
        const ey = ty(f.estimate.cy);
        // 3 sigma: the same 99% region the NIS gate uses to decide whether a
        // returning detection belongs to this target. Drawing the gate the
        // filter actually applies keeps the display and the algorithm honest.
        const sx = Math.max(4, f.estimate.sigmaX * 3) * scale;
        const sy = Math.max(4, f.estimate.sigmaY * 3) * scale;

        ctx.save();
        ctx.translate(ex, ey);
        ctx.rotate(f.estimate.sigmaTheta);

        // Filled uncertainty region, very low alpha — area, not outline, is what
        // conveys "the target is somewhere in here".
        const eg = ctx.createRadialGradient(0, 0, 0, 0, 0, Math.max(sx, sy));
        eg.addColorStop(0, rgba('ember', 0.16));
        eg.addColorStop(1, rgba('ember', 0));
        ctx.fillStyle = eg;
        ctx.beginPath();
        ctx.ellipse(0, 0, sx, sy, 0, 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = rgba('ember', 0.7);
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        ctx.lineDashOffset = reduceMotion ? 0 : -(now / 60) % 10;
        ctx.beginPath();
        ctx.ellipse(0, 0, sx, sy, 0, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore();

        // Dashed ghost brackets at the predicted position.
        ctx.save();
        ctx.setLineDash([4, 4]);
        ctx.shadowColor = rgba('ember', 0.7);
        ctx.shadowBlur = 10;
        ctx.strokeStyle = rgba('ember', 0.9);
        brackets(ex, ey, 34 * scale * 2.1, 34 * scale * 2.1, 8, 1.4);
        ctx.restore();
        ctx.setLineDash([]);

        // Velocity vector from the filter — shows WHERE it expects the target to
        // go, which is the part that justifies driving the servos blind.
        const vlen = Math.hypot(f.estimate.vx, f.estimate.vy);
        if (vlen > 0.5) {
          const k = Math.min(60, vlen * 0.25) / vlen;
          ctx.strokeStyle = rgba('ember', 0.55);
          ctx.lineWidth = 1.4;
          ctx.beginPath();
          ctx.moveTo(ex, ey);
          ctx.lineTo(ex + f.estimate.vx * k * scale, ey + f.estimate.vy * k * scale);
          ctx.stroke();
        }

        ctx.font = '500 10px "IBM Plex Mono", monospace';
        ctx.fillStyle = rgba('ember', 0.9);
        ctx.fillText(
          `PREDICTED · ${f.estimate.coastFrames}f BLIND · 3σ ${(f.estimate.sigmaX * 3).toFixed(0)}px`,
          ex + 14,
          ey + 26,
        );
      }

      /* ---- frame edge: subtle inner vignette so the feed sits in the bezel ---- */
      ctx.strokeStyle = 'rgba(255,255,255,0.06)';
      ctx.lineWidth = 1;
      ctx.strokeRect(ox + 0.5, oy + 0.5, dw - 1, dh - 1);
    };

    raf = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, []);

  return (
    <div
      ref={wrapRef}
      className={`relative h-full w-full overflow-hidden rounded-[1rem] ${className}`}
    >
      {/* The blurred glass bezel: an inset ring that catches light on its top
          edge, so the feed appears recessed behind glass rather than pasted on. */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 z-bezel rounded-[1rem]"
        style={{
          boxShadow:
            'inset 0 0 0 1px rgba(255,255,255,0.10), inset 0 1px 0 0 rgba(255,255,255,0.16), inset 0 -20px 40px -24px rgba(0,0,0,0.9), inset 0 20px 40px -30px rgba(255,255,255,0.10)',
        }}
      />
      <motion.canvas
        ref={canvasRef}
        initial={{ opacity: 0, scale: 1.01 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
        className="block h-full w-full"
      />
    </div>
  );
}
