/**
 * SPECTRA GUARD — telemetry bus.
 *
 * THE PERFORMANCE ARGUMENT, up front, because it dictates this whole file:
 *
 * A 60 Hz telemetry stream must NEVER pass through React state. Calling
 * setState 60 times a second re-runs the component tree, re-runs every hook,
 * and hands React a reconciliation job it cannot finish inside a 16.6 ms frame
 * once the tree is more than trivial. The canvas then misses its vsync and the
 * "60 FPS console" visibly stutters — which on a projector in front of a jury is
 * the one failure they will remember.
 *
 * So telemetry lives here, in mutable fields outside React's world:
 *
 *   - Canvas/rAF consumers (viewport, trajectory plot, confidence ring) read
 *     `bus.latest` directly inside their own animation frame and paint. Zero
 *     renders, zero garbage per frame.
 *   - React components subscribe only to LOW-frequency derived state (PAT stage
 *     changed, link dropped, a new log line) and are notified at a throttled
 *     rate via `subscribeSlow`.
 *
 * The ring buffers are pre-allocated typed arrays. At 60 Hz, allocating a
 * fresh object per sample would hand the GC ~3600 short-lived objects a minute
 * and produce periodic collection pauses that look exactly like dropped frames.
 */

import type { LogEvent, PatStage, RunMeta, TelemetryFrame } from './types';

/** 30 seconds of history at 60 Hz. Enough for the plot, bounded in memory. */
export const HISTORY_CAPACITY = 1800;

/** Max log lines retained. The terminal virtualises, but memory is still finite. */
const LOG_CAPACITY = 500;

/** Slow-subscriber notification interval. 10 Hz is imperceptible for text. */
const SLOW_NOTIFY_MS = 100;

export type LinkMode = 'OFFLINE' | 'LINK' | 'SIM';

/**
 * Snapshot of the state React is allowed to see. Deliberately small: if a field
 * changes at frame rate it does not belong here.
 */
export interface SlowState {
  linkMode: LinkMode;
  stage: PatStage;
  running: boolean;
  detector: string;
  logCount: number;
  /** Bumped whenever meta changes so consumers can re-read intrinsics. */
  metaVersion: number;
}

/**
 * Fixed-capacity ring of float samples. Writes are O(1) with no allocation;
 * reads hand back the raw array plus a cursor so consumers can walk it in
 * chronological order without copying.
 */
class Ring {
  readonly data: Float32Array;
  /** Next write index. */
  cursor = 0;
  /** Total samples ever written — lets consumers detect wrap and staleness. */
  written = 0;

  constructor(capacity: number) {
    this.data = new Float32Array(capacity);
  }

  push(v: number): void {
    this.data[this.cursor] = v;
    this.cursor = (this.cursor + 1) % this.data.length;
    this.written++;
  }

  get length(): number {
    return Math.min(this.written, this.data.length);
  }

  /** Chronological index i (0 = oldest retained) -> physical array index. */
  at(i: number): number {
    const len = this.length;
    const start = this.written <= this.data.length ? 0 : this.cursor;
    return this.data[(start + i) % this.data.length];
  }

  clear(): void {
    this.data.fill(0);
    this.cursor = 0;
    this.written = 0;
  }
}

export interface HistorySeries {
  t: Ring;
  /**
   * Ground-truth beacon position projected into image space.
   *
   * This is the *truth* channel, not the detection channel — it is what makes
   * the trajectory plot an error plot rather than a self-congratulation. The
   * tracker never sees it; it exists only because we own the simulation and can
   * therefore score ourselves honestly.
   */
  truthX: Ring;
  truthY: Ring;
  /** Detector centroid, NaN on frames with no detection (occlusion gaps). */
  measX: Ring;
  measY: Ring;
  /** EKF posterior position in image space. */
  estX: Ring;
  estY: Ring;
  /** Euclidean pixel miss distance. */
  errNorm: Ring;
  confidence: Ring;
  /** 1 while coasting, 0 otherwise — lets the plot shade blind segments. */
  coasting: Ring;
}

class TelemetryBus {
  /** The single most recent frame. Canvas consumers read this and only this. */
  latest: TelemetryFrame | null = null;

  /** Camera intrinsics and run identity. */
  meta: RunMeta | null = null;

  /**
   * Latest decoded camera image. An ImageBitmap (not a data URL) because
   * `drawImage` of an ImageBitmap is a straight GPU texture upload, whereas a
   * data URL forces a decode on the main thread every single frame.
   */
  image: ImageBitmap | HTMLCanvasElement | null = null;

  /** Seq of the frame `image` belongs to, for pairing/staleness checks. */
  imageSeq = -1;

  readonly history: HistorySeries = {
    t: new Ring(HISTORY_CAPACITY),
    truthX: new Ring(HISTORY_CAPACITY),
    truthY: new Ring(HISTORY_CAPACITY),
    measX: new Ring(HISTORY_CAPACITY),
    measY: new Ring(HISTORY_CAPACITY),
    estX: new Ring(HISTORY_CAPACITY),
    estY: new Ring(HISTORY_CAPACITY),
    errNorm: new Ring(HISTORY_CAPACITY),
    confidence: new Ring(HISTORY_CAPACITY),
    coasting: new Ring(HISTORY_CAPACITY),
  };

  logs: LogEvent[] = [];

  private slow: SlowState = {
    linkMode: 'OFFLINE',
    stage: 'IDLE',
    running: false,
    detector: '—',
    logCount: 0,
    metaVersion: 0,
  };

  private listeners = new Set<(s: SlowState) => void>();
  private lastNotify = 0;
  private notifyQueued = false;

  /**
   * Cached camera intrinsics, recomputed only when `meta` arrives.
   *
   * f_px = (width/2) / tan(FOV/2) is the pinhole focal length in pixels. We keep
   * it here rather than recomputing per frame because the projection below runs
   * 60 times a second and a `tan` call is not free.
   */
  private fPx = (640 / 2) / Math.tan((8.0 * Math.PI) / 180 / 2);
  private cx0 = 320;
  private cy0 = 240;

  /* ------------------------------ ingestion ----------------------------- */

  pushFrame(frame: TelemetryFrame): void {
    this.latest = frame;

    // Project ground truth into image space with the same pinhole model the EKF
    // linearises. tan(), not the small-angle approximation: off-boresight during
    // SEARCH the two disagree by several pixels, and a truth trace that is
    // itself several pixels wrong would flatter the tracker.
    const dAz = ((frame.truth.azTrue - frame.gimbal.pan) * Math.PI) / 180;
    const dEl = ((frame.truth.elTrue - frame.gimbal.tilt) * Math.PI) / 180;
    // Guard the ±90° singularity: during a lost-target SEARCH the truth can sit
    // outside the frustum entirely, where tan() explodes.
    const inFrustum = Math.abs(dAz) < 1.2 && Math.abs(dEl) < 1.2;
    const truthU = inFrustum ? this.cx0 + this.fPx * Math.tan(dAz) : Number.NaN;
    const truthV = inFrustum ? this.cy0 - this.fPx * Math.tan(dEl) : Number.NaN;

    const h = this.history;
    h.t.push(frame.t);
    h.truthX.push(truthU);
    h.truthY.push(truthV);
    h.measX.push(frame.detection ? frame.detection.cx : Number.NaN);
    h.measY.push(frame.detection ? frame.detection.cy : Number.NaN);
    h.estX.push(frame.estimate.cx);
    h.estY.push(frame.estimate.cy);
    h.errNorm.push(frame.error.norm);
    h.confidence.push(frame.confidence);
    h.coasting.push(frame.stage === 'COAST' ? 1 : 0);

    // Stage transitions are rare and semantically important, so they bypass the
    // throttle and notify React immediately — a jury watching COAST engage
    // should see the UI react on the same frame, not up to 100 ms later.
    if (frame.stage !== this.slow.stage) {
      this.slow = { ...this.slow, stage: frame.stage };
      this.flush();
    } else {
      this.scheduleNotify();
    }
  }

  pushLog(log: LogEvent): void {
    this.logs.push(log);
    if (this.logs.length > LOG_CAPACITY) {
      this.logs.splice(0, this.logs.length - LOG_CAPACITY);
    }
    this.slow = { ...this.slow, logCount: this.logs.length };
    this.scheduleNotify();
  }

  setMeta(meta: RunMeta): void {
    this.meta = meta;
    this.fPx = meta.width / 2 / Math.tan((meta.fovDeg * Math.PI) / 180 / 2);
    this.cx0 = meta.width / 2;
    this.cy0 = meta.height / 2;
    this.slow = {
      ...this.slow,
      detector: meta.detector,
      metaVersion: this.slow.metaVersion + 1,
    };
    this.flush();
  }

  setImage(img: ImageBitmap | HTMLCanvasElement, seq: number): void {
    // Release the previous bitmap explicitly. ImageBitmaps hold GPU-backed
    // memory that the GC frees lazily; at 60 Hz "lazily" means hundreds of
    // megabytes of stale textures before the collector gets around to it.
    if (this.image && 'close' in this.image && typeof this.image.close === 'function') {
      this.image.close();
    }
    this.image = img;
    this.imageSeq = seq;
  }

  setLinkMode(mode: LinkMode): void {
    if (this.slow.linkMode === mode) return;
    this.slow = { ...this.slow, linkMode: mode };
    this.flush();
  }

  setRunning(running: boolean): void {
    if (this.slow.running === running) return;
    this.slow = { ...this.slow, running };
    this.flush();
  }

  reset(): void {
    this.latest = null;
    this.imageSeq = -1;
    if (this.image && 'close' in this.image && typeof this.image.close === 'function') {
      this.image.close();
    }
    this.image = null;
    Object.values(this.history).forEach((r) => (r as Ring).clear());
    this.logs = [];
    this.slow = { ...this.slow, stage: 'IDLE', running: false, logCount: 0 };
    this.flush();
  }

  /* ----------------------------- subscription --------------------------- */

  getSlowState = (): SlowState => this.slow;

  subscribeSlow = (cb: (s: SlowState) => void): (() => void) => {
    this.listeners.add(cb);
    return () => this.listeners.delete(cb);
  };

  /**
   * Coalesce notifications to SLOW_NOTIFY_MS. Without this, a burst of log
   * events would each trigger a React render and we would have reintroduced
   * exactly the problem this class exists to avoid.
   */
  private scheduleNotify(): void {
    if (this.notifyQueued) return;
    const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
    const wait = Math.max(0, SLOW_NOTIFY_MS - (now - this.lastNotify));
    this.notifyQueued = true;
    setTimeout(() => {
      this.notifyQueued = false;
      this.flush();
    }, wait);
  }

  private flush(): void {
    this.lastNotify = typeof performance !== 'undefined' ? performance.now() : Date.now();
    this.listeners.forEach((cb) => cb(this.slow));
  }

  /* ------------------------------- export ------------------------------- */

  /**
   * The performance log the problem statement asks us to deliver, as CSV.
   * Built from the ring buffers rather than a separate accumulator so what gets
   * exported is provably the same data the operator was looking at.
   */
  toCsv(): string {
    const h = this.history;
    const m = this.meta;
    const rows: string[] = [];
    // A header block, because a bare CSV with no provenance is not evidence.
    // Anyone re-deriving our numbers needs the run identity and the intrinsics
    // the pixel columns are expressed in.
    rows.push(`# SPECTRA GUARD performance log`);
    rows.push(`# run_id,${m?.runId ?? 'UNKNOWN'}`);
    rows.push(`# trajectory,${m?.trajectory ?? 'UNKNOWN'}`);
    rows.push(`# detector,${m?.detector ?? 'UNKNOWN'}`);
    rows.push(`# fov_deg,${m?.fovDeg ?? ''},width_px,${m?.width ?? ''},height_px,${m?.height ?? ''}`);
    rows.push(`# seed,${m?.seed ?? ''},target_hz,${m?.targetHz ?? ''}`);
    rows.push(`# exported_utc,${new Date().toISOString()}`);
    rows.push(
      't_s,truth_x_px,truth_y_px,meas_x_px,meas_y_px,est_x_px,est_y_px,error_px,confidence,coasting',
    );
    const n = h.t.length;
    for (let i = 0; i < n; i++) {
      rows.push(
        [
          h.t.at(i).toFixed(4),
          fmtCsv(h.truthX.at(i)),
          fmtCsv(h.truthY.at(i)),
          fmtCsv(h.measX.at(i)),
          fmtCsv(h.measY.at(i)),
          h.estX.at(i).toFixed(3),
          h.estY.at(i).toFixed(3),
          h.errNorm.at(i).toFixed(3),
          h.confidence.at(i).toFixed(4),
          h.coasting.at(i).toFixed(0),
        ].join(','),
      );
    }
    return rows.join('\n');
  }
}

function fmtCsv(v: number): string {
  return Number.isFinite(v) ? v.toFixed(3) : '';
}

/**
 * Module-level singleton. There is exactly one camera and one PAT loop, so a
 * single bus is the honest model; passing it through context would only add
 * indirection and tempt someone into putting it in state.
 */
export const bus = new TelemetryBus();
export type { TelemetryBus };
