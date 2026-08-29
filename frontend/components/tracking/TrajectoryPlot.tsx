'use client';

/**
 * SPECTRA GUARD — Trajectory & Residual plot.
 *
 * Two linked views on one canvas, one animation frame:
 *
 *   TOP — the path. Ground truth projected into image space (thin, neutral)
 *   against the EKF posterior (bright, glowing). Where the detector went blind
 *   the estimate turns amber, so a jury can see the exact segment where the
 *   system was flying on prediction alone and watch it rejoin the truth line
 *   afterwards. That rejoining is the whole claim of the project, and it is far
 *   more persuasive as a picture than as a number.
 *
 *   BOTTOM — the residual. |truth − estimate| over time against the lock
 *   tolerance, with coast spans shaded. This is the error plot the technical
 *   report needs, drawn live from the same ring buffers the CSV exports.
 *
 * ONE canvas rather than two: two canvases means two rAF loops, two composited
 * layers over a blurred backdrop, and two chances to tear. Splitting a single
 * canvas into regions costs nothing and keeps the two views frame-locked, which
 * matters because the operator reads them together.
 *
 * Reads `bus.history` directly — no React state, no data copying. The rings are
 * Float32Arrays; walking them is a pointer chase, not an allocation.
 */

import { motion } from 'framer-motion';
import { useEffect, useRef } from 'react';
import { bus } from '@/lib/telemetry-bus';
import { LOCK_TOLERANCE_PX } from '@/lib/types';

/** Samples shown. 900 at 60 Hz = 15 s — long enough to contain a full occlusion. */
const WINDOW = 900;

/** Minimum half-extent of the XY view, in pixels of image space. */
const MIN_HALF_SPAN = 40;

export function TrajectoryPlot({ className = '' }: { className?: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let cssW = 0;
    let cssH = 0;
    const resize = () => {
      const rect = wrap.getBoundingClientRect();
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

    /**
     * Eased view bounds.
     *
     * Recomputing min/max per frame and using it directly makes the whole plot
     * twitch every time a single sample enters or leaves the window — the data
     * looks noisier than it is. Easing the bounds toward the target means the
     * camera pans smoothly and the trace itself carries the motion.
     */
    let vcx = 320;
    let vcy = 240;
    let vspan = 120;
    let vErrMax = LOCK_TOLERANCE_PX * 3;
    let initialised = false;

    let raf = 0;

    const draw = (now: number) => {
      raf = requestAnimationFrame(draw);
      if (cssW < 2 || cssH < 2) return;

      ctx.clearRect(0, 0, cssW, cssH);

      const h = bus.history;
      const n = h.t.length;

      /* ------------------------------ layout ----------------------------- */
      const PAD = 10;
      const GAP = 14;
      const errH = Math.max(46, Math.min(96, cssH * 0.3));
      const pathTop = PAD;
      const pathH = cssH - errH - GAP - PAD * 2;
      const pathBot = pathTop + pathH;
      const errTop = pathBot + GAP;
      const plotL = PAD + 26; // room for the residual axis labels
      const plotR = cssW - PAD;
      const plotW = plotR - plotL;

      if (n < 2 || pathH < 20) {
        ctx.fillStyle = 'rgba(92,110,133,0.75)';
        ctx.font = '500 10px "IBM Plex Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText('AWAITING TELEMETRY', cssW / 2, cssH / 2);
        ctx.textAlign = 'left';
        return;
      }

      const from = Math.max(0, n - WINDOW);
      const count = n - from;

      /* --------------------- bounds over the visible window --------------- */
      let minX = Infinity;
      let maxX = -Infinity;
      let minY = Infinity;
      let maxY = -Infinity;
      let errMax = LOCK_TOLERANCE_PX * 1.5;
      for (let i = from; i < n; i++) {
        const ex = h.estX.at(i);
        const ey = h.estY.at(i);
        if (Number.isFinite(ex)) {
          if (ex < minX) minX = ex;
          if (ex > maxX) maxX = ex;
        }
        if (Number.isFinite(ey)) {
          if (ey < minY) minY = ey;
          if (ey > maxY) maxY = ey;
        }
        const tx = h.truthX.at(i);
        const tyv = h.truthY.at(i);
        if (Number.isFinite(tx)) {
          if (tx < minX) minX = tx;
          if (tx > maxX) maxX = tx;
        }
        if (Number.isFinite(tyv)) {
          if (tyv < minY) minY = tyv;
          if (tyv > maxY) maxY = tyv;
        }
        const e = h.errNorm.at(i);
        if (Number.isFinite(e) && e > errMax) errMax = e;
      }
      if (!Number.isFinite(minX)) return;

      const tcx = (minX + maxX) / 2;
      const tcy = (minY + maxY) / 2;
      // Square aspect: an unequal x/y scale would distort the shape of the path
      // and make an isotropic error look directional.
      const tspan = Math.max(MIN_HALF_SPAN, (Math.max(maxX - minX, maxY - minY) / 2) * 1.25);

      if (!initialised) {
        vcx = tcx;
        vcy = tcy;
        vspan = tspan;
        vErrMax = errMax;
        initialised = true;
      } else {
        const k = 0.06;
        vcx += (tcx - vcx) * k;
        vcy += (tcy - vcy) * k;
        vspan += (tspan - vspan) * k;
        // Asymmetric: expand fast so a spike is never clipped off the top, relax
        // slowly so the axis does not breathe.
        vErrMax += (errMax - vErrMax) * (errMax > vErrMax ? 0.25 : 0.02);
      }

      const scale = Math.min(plotW, pathH) / (vspan * 2);
      const pcx = plotL + plotW / 2;
      const pcy = pathTop + pathH / 2;
      const PX = (x: number) => pcx + (x - vcx) * scale;
      const PY = (y: number) => pcy + (y - vcy) * scale;

      /* -------------------------- path grid ------------------------------ */
      ctx.save();
      ctx.beginPath();
      ctx.rect(plotL, pathTop, plotW, pathH);
      ctx.clip();

      // Grid spacing chosen in image-space pixels, snapped to a 1/2/5 decade so
      // the label is always a round number the eye can use as a ruler.
      const rawStep = (vspan * 2) / 5;
      const decade = Math.pow(10, Math.floor(Math.log10(rawStep)));
      const mult = rawStep / decade;
      const step = decade * (mult < 1.5 ? 1 : mult < 3.5 ? 2 : mult < 7.5 ? 5 : 10);

      ctx.strokeStyle = 'rgba(255,255,255,0.045)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      const gx0 = Math.ceil((vcx - vspan) / step) * step;
      for (let x = gx0; x <= vcx + vspan; x += step) {
        const sx = Math.round(PX(x)) + 0.5;
        ctx.moveTo(sx, pathTop);
        ctx.lineTo(sx, pathBot);
      }
      const gy0 = Math.ceil((vcy - vspan) / step) * step;
      for (let y = gy0; y <= vcy + vspan; y += step) {
        const sy = Math.round(PY(y)) + 0.5;
        ctx.moveTo(plotL, sy);
        ctx.lineTo(plotR, sy);
      }
      ctx.stroke();

      // Boresight: the point the controller is trying to drive the target onto.
      const bxs = PX(bus.meta ? bus.meta.width / 2 : 320);
      const bys = PY(bus.meta ? bus.meta.height / 2 : 240);
      ctx.strokeStyle = 'rgba(232,238,247,0.22)';
      ctx.setLineDash([2, 4]);
      ctx.beginPath();
      ctx.moveTo(bxs - 8, bys);
      ctx.lineTo(bxs + 8, bys);
      ctx.moveTo(bxs, bys - 8);
      ctx.lineTo(bxs, bys + 8);
      ctx.stroke();
      ctx.setLineDash([]);

      /* ---------------------- truth path (reference) ---------------------- */
      ctx.strokeStyle = 'rgba(232,238,247,0.34)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      let pen = false;
      for (let i = from; i < n; i++) {
        const x = h.truthX.at(i);
        const y = h.truthY.at(i);
        if (!Number.isFinite(x) || !Number.isFinite(y)) {
          pen = false;
          continue;
        }
        const sx = PX(x);
        const sy = PY(y);
        if (!pen) {
          ctx.moveTo(sx, sy);
          pen = true;
        } else ctx.lineTo(sx, sy);
      }
      ctx.stroke();

      /* --------- estimate path, coloured by whether we could see ---------- */
      // Drawn as consecutive runs of the same state so each run is one stroked
      // path. Per-segment strokes would be `count` separate draw calls.
      ctx.lineWidth = 1.8;
      ctx.lineJoin = 'round';
      let runStart = from;
      const stateAt = (i: number) => (h.coasting.at(i) > 0.5 ? 1 : 0);
      let runState = stateAt(from);
      const strokeRun = (a: number, b: number, blind: number) => {
        if (b - a < 1) return;
        ctx.save();
        if (blind) {
          ctx.strokeStyle = 'rgba(255,169,43,0.95)';
          ctx.shadowColor = 'rgba(255,169,43,0.8)';
        } else {
          ctx.strokeStyle = 'rgba(34,224,255,0.95)';
          ctx.shadowColor = 'rgba(34,224,255,0.7)';
        }
        ctx.shadowBlur = 8;
        ctx.beginPath();
        for (let i = a; i <= b; i++) {
          const sx = PX(h.estX.at(i));
          const sy = PY(h.estY.at(i));
          if (i === a) ctx.moveTo(sx, sy);
          else ctx.lineTo(sx, sy);
        }
        ctx.stroke();
        ctx.restore();
      };
      for (let i = from + 1; i < n; i++) {
        const s = stateAt(i);
        if (s !== runState) {
          // Overlap by one sample so the runs join without a visible seam.
          strokeRun(runStart, i, runState);
          runStart = i - 1;
          runState = s;
        }
      }
      strokeRun(runStart, n - 1, runState);

      /* ------------------------- head of the trace ----------------------- */
      const hx = PX(h.estX.at(n - 1));
      const hy = PY(h.estY.at(n - 1));
      const blindNow = stateAt(n - 1) === 1;
      const rgb = blindNow ? '255,169,43' : '34,224,255';

      // Subsurface radial bloom: light coming from behind the glass at the
      // current estimate, rather than a sticker drawn on top of it.
      const bloom = ctx.createRadialGradient(hx, hy, 0, hx, hy, 34);
      bloom.addColorStop(0, `rgba(${rgb},0.34)`);
      bloom.addColorStop(0.5, `rgba(${rgb},0.10)`);
      bloom.addColorStop(1, `rgba(${rgb},0)`);
      ctx.fillStyle = bloom;
      ctx.fillRect(hx - 34, hy - 34, 68, 68);

      ctx.save();
      ctx.shadowColor = `rgba(${rgb},1)`;
      ctx.shadowBlur = 12;
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(hx, hy, 2.6, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      // Truth head, so the instantaneous gap between belief and reality is a
      // visible distance rather than something to be inferred.
      const thx = h.truthX.at(n - 1);
      const thy = h.truthY.at(n - 1);
      if (Number.isFinite(thx) && Number.isFinite(thy)) {
        const sx = PX(thx);
        const sy = PY(thy);
        ctx.strokeStyle = 'rgba(232,238,247,0.8)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(sx, sy, 3.4, 0, Math.PI * 2);
        ctx.stroke();
        // Connector: the residual, drawn as the segment it actually is.
        if (Math.hypot(sx - hx, sy - hy) > 5) {
          ctx.strokeStyle = 'rgba(232,238,247,0.28)';
          ctx.setLineDash([2, 3]);
          ctx.beginPath();
          ctx.moveTo(sx, sy);
          ctx.lineTo(hx, hy);
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }
      ctx.restore(); // un-clip

      /* --------------------------- scale bar ----------------------------- */
      const barPx = step * scale;
      ctx.strokeStyle = 'rgba(232,238,247,0.4)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(plotL + 4, pathBot - 6);
      ctx.lineTo(plotL + 4 + barPx, pathBot - 6);
      ctx.moveTo(plotL + 4, pathBot - 9);
      ctx.lineTo(plotL + 4, pathBot - 3);
      ctx.moveTo(plotL + 4 + barPx, pathBot - 9);
      ctx.lineTo(plotL + 4 + barPx, pathBot - 3);
      ctx.stroke();
      ctx.font = '500 9px "IBM Plex Mono", monospace';
      ctx.fillStyle = 'rgba(148,166,188,0.85)';
      ctx.fillText(`${step >= 1 ? step.toFixed(0) : step.toFixed(1)} px`, plotL + 8 + barPx, pathBot - 3);

      /* ======================== residual strip =========================== */
      const eBot = errTop + errH;
      const EY = (e: number) => eBot - (Math.min(e, vErrMax) / vErrMax) * errH;
      const EX = (i: number) => plotL + ((i - from) / Math.max(1, count - 1)) * plotW;

      // Coast spans shaded first, so the trace draws over them.
      let spanStart = -1;
      for (let i = from; i < n; i++) {
        const blind = h.coasting.at(i) > 0.5;
        if (blind && spanStart < 0) spanStart = i;
        if ((!blind || i === n - 1) && spanStart >= 0) {
          const x0 = EX(spanStart);
          const x1 = EX(i);
          ctx.fillStyle = 'rgba(255,169,43,0.10)';
          ctx.fillRect(x0, errTop, Math.max(1, x1 - x0), errH);
          spanStart = -1;
        }
      }

      // The in-spec band. Drawing the acceptance region as a filled band rather
      // than a single line lets the operator see margin, not just pass/fail.
      const tolY = EY(LOCK_TOLERANCE_PX);
      ctx.fillStyle = 'rgba(34,224,255,0.06)';
      ctx.fillRect(plotL, tolY, plotW, eBot - tolY);
      ctx.strokeStyle = 'rgba(34,224,255,0.4)';
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(plotL, tolY + 0.5);
      ctx.lineTo(plotR, tolY + 0.5);
      ctx.stroke();
      ctx.setLineDash([]);

      // Axis frame and labels.
      ctx.strokeStyle = 'rgba(255,255,255,0.08)';
      ctx.beginPath();
      ctx.moveTo(plotL, errTop);
      ctx.lineTo(plotL, eBot);
      ctx.lineTo(plotR, eBot);
      ctx.stroke();
      ctx.font = '500 8px "IBM Plex Mono", monospace';
      ctx.fillStyle = 'rgba(92,110,133,0.9)';
      ctx.textAlign = 'right';
      ctx.fillText(vErrMax.toFixed(0), plotL - 4, errTop + 7);
      ctx.fillText('0', plotL - 4, eBot);
      ctx.fillStyle = 'rgba(34,224,255,0.75)';
      ctx.fillText(String(LOCK_TOLERANCE_PX), plotL - 4, tolY + 3);
      ctx.textAlign = 'left';

      // Residual trace, filled under the curve so magnitude reads as area.
      ctx.beginPath();
      ctx.moveTo(plotL, eBot);
      for (let i = from; i < n; i++) {
        ctx.lineTo(EX(i), EY(h.errNorm.at(i)));
      }
      ctx.lineTo(EX(n - 1), eBot);
      ctx.closePath();
      const fill = ctx.createLinearGradient(0, errTop, 0, eBot);
      fill.addColorStop(0, 'rgba(34,224,255,0.22)');
      fill.addColorStop(1, 'rgba(34,224,255,0.01)');
      ctx.fillStyle = fill;
      ctx.fill();

      ctx.save();
      ctx.shadowColor = 'rgba(34,224,255,0.55)';
      ctx.shadowBlur = 6;
      ctx.strokeStyle = 'rgba(34,224,255,0.9)';
      ctx.lineWidth = 1.3;
      ctx.beginPath();
      for (let i = from; i < n; i++) {
        const x = EX(i);
        const y = EY(h.errNorm.at(i));
        if (i === from) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.restore();

      // Live edge marker on the residual, tied to the same instant as the path
      // head above — the two views share a cursor.
      const lastE = h.errNorm.at(n - 1);
      ctx.fillStyle = lastE <= LOCK_TOLERANCE_PX ? 'rgb(34,224,255)' : 'rgb(255,169,43)';
      ctx.beginPath();
      ctx.arc(EX(n - 1), EY(lastE), 2, 0, Math.PI * 2);
      ctx.fill();

      void now;
    };

    raf = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, []);

  return (
    <div ref={wrapRef} className={`relative h-full w-full ${className}`}>
      <motion.canvas
        ref={canvasRef}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1], delay: 0.15 }}
        className="block h-full w-full"
      />
      {/* Legend as real text, not baked into the canvas: it is static, and text
          nodes stay selectable and accessible. */}
      <div className="pointer-events-none absolute right-3 top-2 flex items-center gap-3">
        <LegendKey color="rgba(232,238,247,0.55)" label="truth" dashed />
        <LegendKey color="rgb(34,224,255)" label="ekf" />
        <LegendKey color="rgb(255,169,43)" label="coast" />
      </div>
    </div>
  );
}

function LegendKey({
  color,
  label,
  dashed = false,
}: {
  color: string;
  label: string;
  dashed?: boolean;
}) {
  return (
    <span className="flex items-center gap-1.5">
      <span
        className="block h-0 w-3.5"
        style={{
          borderTopWidth: dashed ? 1 : 2,
          borderTopStyle: dashed ? 'dashed' : 'solid',
          borderTopColor: color,
        }}
      />
      <span className="font-mono text-[0.5rem] uppercase tracking-[0.14em] text-readout-tertiary">
        {label}
      </span>
    </span>
  );
}
