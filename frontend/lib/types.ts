/**
 * SPECTRA GUARD — shared wire types.
 *
 * This file is the single source of truth for the backend<->frontend contract.
 * The Python side mirrors it in `backend/spectraguard/schema.py`; if you change
 * a field here, change it there in the same commit.
 *
 * Transport (see docs/PROTOCOL.md):
 *   - WebSocket TEXT frames  -> JSON `TelemetryFrame` / `ServerEvent`, ~60 Hz
 *   - WebSocket BINARY frames -> raw JPEG camera image for the current `seq`
 *   Telemetry is sent immediately BEFORE its matching image so the renderer can
 *   pair them without a header, and drop stale images under backpressure.
 */

/** The PAT state machine. Order is meaningful: it is the acquisition sequence. */
export type PatStage =
  | 'IDLE' // engine loaded, no target commanded
  | 'SEARCH' // spiral/raster scan of the uncertainty volume
  | 'ACQUIRE' // beacon seen, EKF initialising, not yet inside lock tolerance
  | 'TRACK' // closed-loop lock, error inside tolerance
  | 'COAST' // beacon lost, driving gimbal off EKF prediction alone
  | 'FAULT'; // unrecoverable — coast timeout exceeded, returns to SEARCH

export const PAT_STAGES: readonly PatStage[] = [
  'IDLE',
  'SEARCH',
  'ACQUIRE',
  'TRACK',
  'COAST',
  'FAULT',
] as const;

/** Target motion models available to the scene generator. */
export type TrajectoryKind =
  | 'LEO_SATELLITE' // high angular rate, smooth, predictable pass
  | 'UAV_ORBIT' // slow loiter circle with wind-driven perturbation
  | 'UAV_EVASIVE' // aggressive jinking, defeats naive predictors
  | 'STATIC_TEST'; // boresight calibration target

export type TunerMode = 'FIXED' | 'SAC';

/** Detector backend actually in use this run. */
export type DetectorBackend = 'YOLOV11' | 'CLASSICAL';

/** Gimbal pointing state. Angles in degrees, rates in degrees/second. */
export interface GimbalState {
  pan: number;
  tilt: number;
  panRate: number;
  tiltRate: number;
}

/** Ground-truth target bearing. Simulation only — never available to the tracker. */
export interface TargetTruth {
  azTrue: number;
  elTrue: number;
  rangeKm: number;
}

/** A single detection in image space. Pixels, origin top-left. */
export interface Detection {
  cx: number;
  cy: number;
  w: number;
  h: number;
  /** Detector confidence, 0..1. */
  score: number;
  /** Sub-pixel centroid refinement applied on top of the box centre. */
  subpixel: boolean;
}

/**
 * EKF posterior. `sigmaX`/`sigmaY`/`sigmaTheta` describe the 1-sigma position
 * error ellipse in pixels and its rotation in radians — this drives the ghost
 * reticle in the viewport.
 */
export interface EkfEstimate {
  cx: number;
  cy: number;
  vx: number;
  vy: number;
  sigmaX: number;
  sigmaY: number;
  sigmaTheta: number;
  /** Frames since the last accepted measurement. 0 while tracking. */
  coastFrames: number;
}

/** Closed-loop error, in image space and reflected back to angle space. */
export interface TrackingError {
  px: number;
  py: number;
  /** Euclidean pixel miss distance from boresight. */
  norm: number;
  azErr: number;
  elErr: number;
}

/** The performance log the problem statement asks us to deliver. */
export interface PerformanceMetrics {
  fps: number;
  /** Milliseconds from SEARCH entry to first TRACK entry. Null until acquired. */
  acqTimeMs: number | null;
  /** Fraction of frames since acquisition spent in TRACK, 0..1. */
  lockRetention: number;
  rmsErrorPx: number;
  /** Normalised innovation squared — the EKF's own consistency check. */
  nis: number;
  /** Wall-clock ms spent in detect+estimate+control for this frame. */
  computeMs: number;
}

/** Operator-injected physical disturbances. */
export interface Disturbance {
  /** Refractive index structure constant, m^(-2/3). 1e-17 (calm) .. 1e-13 (severe). */
  cn2: number;
  /** Dominant platform vibration frequency, Hz. */
  jitterHz: number;
  /** Platform vibration amplitude, microradians RMS. */
  jitterUrad: number;
  /** Additive white Gaussian noise sigma in DN (8-bit counts). */
  awgnSigma: number;
  /** Cloud/obscuration opacity, 0..1. Drives COAST transitions. */
  occlusion: number;
}

/** Controller state, including gains the SAC agent may be rewriting live. */
export interface ControlState {
  kp: number;
  ki: number;
  kd: number;
  tunerMode: TunerMode;
  /** Commanded rate this frame, deg/s. */
  effortPan: number;
  effortTilt: number;
  /** SAC critic value estimate — shown as the tuner's own confidence. */
  criticValue: number | null;
}

/** One tick of the PAT loop. Everything the UI renders comes from here. */
export interface TelemetryFrame {
  kind: 'telemetry';
  seq: number;
  /** Simulation time, seconds since run start. */
  t: number;
  stage: PatStage;
  gimbal: GimbalState;
  truth: TargetTruth;
  detection: Detection | null;
  estimate: EkfEstimate;
  error: TrackingError;
  /** Fused lock confidence, 0..1. Detector score gated by EKF consistency. */
  confidence: number;
  metrics: PerformanceMetrics;
  disturbance: Disturbance;
  control: ControlState;
  /** True Strehl-like intensity attenuation from scintillation this frame, 0..1. */
  scintillation: number;
}

export type LogLevel = 'INFO' | 'WARN' | 'ERROR' | 'OK';

/** Discrete events worth printing to the terminal. */
export interface LogEvent {
  kind: 'log';
  seq: number;
  t: number;
  level: LogLevel;
  /** Short channel tag, e.g. 'EKF', 'PAT', 'SAC', 'OPT'. */
  channel: string;
  message: string;
}

export interface RunMeta {
  kind: 'meta';
  runId: string;
  trajectory: TrajectoryKind;
  detector: DetectorBackend;
  /** Camera intrinsics the frontend needs to map pixels to angles. */
  fovDeg: number;
  width: number;
  height: number;
  /** Loop target rate, Hz. */
  targetHz: number;
  seed: number;
}

export type ServerEvent = TelemetryFrame | LogEvent | RunMeta;

/* ------------------------------- commands -------------------------------- */

export type ClientCommand =
  | { type: 'start'; trajectory: TrajectoryKind; seed?: number }
  | { type: 'stop' }
  | { type: 'reset' }
  | { type: 'set_disturbance'; patch: Partial<Disturbance> }
  | { type: 'set_control'; patch: Partial<Pick<ControlState, 'kp' | 'ki' | 'kd' | 'tunerMode'>> }
  | { type: 'set_trajectory'; trajectory: TrajectoryKind }
  | { type: 'nudge'; pan: number; tilt: number }
  | { type: 'export_log' };

/* ------------------------------- defaults -------------------------------- */

export const DEFAULT_DISTURBANCE: Disturbance = {
  cn2: 1e-15,
  jitterHz: 12,
  // 600 urad RMS ~ 2.7 px at this focal length. Vehicle and drone mounts really
  // do vibrate at the milliradian scale, and platform jitter — not turbulence —
  // is what dominates coarse pointing error. See lib/mock-engine.ts header.
  jitterUrad: 600,
  awgnSigma: 4,
  occlusion: 0,
};

export const DEFAULT_CONTROL: Pick<ControlState, 'kp' | 'ki' | 'kd' | 'tunerMode'> = {
  kp: 0.42,
  ki: 0.06,
  kd: 0.11,
  tunerMode: 'SAC',
};

/** Pixel miss distance below which we declare lock. Mirrors backend config. */
export const LOCK_TOLERANCE_PX = 12;

/** Consecutive coast frames after which the run is declared FAULT. */
export const COAST_TIMEOUT_FRAMES = 90;
