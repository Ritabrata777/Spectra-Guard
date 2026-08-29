"""SPECTRA GUARD — wire schema.

This module is the Python half of the contract in `frontend/lib/types.ts`.
`tests/test_schema_parity.py` parses that TypeScript file and asserts that every
interface field appears here with the same name, so the two cannot silently drift
apart. Contract drift between a Python producer and a TypeScript consumer is the
single most common way a project like this dies quietly two days before a demo:
nothing errors, a panel just reads "—" forever.

A NOTE ON NAMING. The dataclass fields below are camelCase, which is not
idiomatic Python. That is deliberate and it is the lesser of two evils. The
alternative is snake_case fields plus a translation layer on the hot path, which
means every field exists under two names, the mapping is a third thing to keep in
sync, and the 60 Hz serialiser does dictionary-key rewriting sixty times a
second. Naming these structures after the wire format they exist to produce keeps
one name per field and makes the parity test a direct comparison.

SERIALISATION. `RunMeta` and `LogEvent` are rare, so they use `dataclasses.asdict`
for clarity. `TelemetryFrame` is emitted 60 times a second, and `asdict` is a
recursive deep-copy that allocates a new dict for every nested object — so it
builds its payload by hand in `wire()`. That is the only place in this codebase
where speed is allowed to beat elegance, and it is measurably worth it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional


PatStage = Literal["IDLE", "SEARCH", "ACQUIRE", "TRACK", "COAST", "FAULT"]
TrajectoryKind = Literal["LEO_SATELLITE", "UAV_ORBIT", "UAV_EVASIVE", "STATIC_TEST"]
TunerMode = Literal["FIXED", "SAC"]
DetectorBackend = Literal["YOLOV11", "CLASSICAL"]
LogLevel = Literal["INFO", "WARN", "ERROR", "OK"]

PAT_STAGES: tuple[str, ...] = ("IDLE", "SEARCH", "ACQUIRE", "TRACK", "COAST", "FAULT")


@dataclass
class GimbalState:
    """Pointing state. Degrees and degrees/second — wire units."""

    pan: float = 0.0
    tilt: float = 0.0
    panRate: float = 0.0
    tiltRate: float = 0.0


@dataclass
class TargetTruth:
    """Ground truth. Simulation only; the tracker never reads this."""

    azTrue: float = 0.0
    elTrue: float = 0.0
    rangeKm: float = 0.0


@dataclass
class Detection:
    cx: float = 0.0
    cy: float = 0.0
    w: float = 0.0
    h: float = 0.0
    score: float = 0.0
    subpixel: bool = False


@dataclass
class EkfEstimate:
    cx: float = 0.0
    cy: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    sigmaX: float = 0.0
    sigmaY: float = 0.0
    sigmaTheta: float = 0.0
    coastFrames: int = 0


@dataclass
class TrackingError:
    px: float = 0.0
    py: float = 0.0
    norm: float = 0.0
    azErr: float = 0.0
    elErr: float = 0.0


@dataclass
class PerformanceMetrics:
    """The performance log the problem statement asks us to deliver."""

    fps: float = 0.0
    acqTimeMs: Optional[float] = None
    lockRetention: float = 0.0
    rmsErrorPx: float = 0.0
    nis: float = 0.0
    computeMs: float = 0.0


@dataclass
class Disturbance:
    cn2: float = 1e-15
    jitterHz: float = 12.0
    jitterUrad: float = 600.0
    awgnSigma: float = 4.0
    occlusion: float = 0.0


@dataclass
class ControlState:
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    tunerMode: str = "FIXED"
    effortPan: float = 0.0
    effortTilt: float = 0.0
    criticValue: Optional[float] = None


@dataclass
class TelemetryFrame:
    seq: int = 0
    t: float = 0.0
    stage: str = "IDLE"
    gimbal: GimbalState = field(default_factory=GimbalState)
    truth: TargetTruth = field(default_factory=TargetTruth)
    detection: Optional[Detection] = None
    estimate: EkfEstimate = field(default_factory=EkfEstimate)
    error: TrackingError = field(default_factory=TrackingError)
    confidence: float = 0.0
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    disturbance: Disturbance = field(default_factory=Disturbance)
    control: ControlState = field(default_factory=ControlState)
    scintillation: float = 1.0

    kind: str = "telemetry"

    def wire(self) -> dict[str, Any]:
        """Hand-built payload. See the module docstring for why not asdict()."""
        g, tr, e, er, m, d, c = (
            self.gimbal,
            self.truth,
            self.estimate,
            self.error,
            self.metrics,
            self.disturbance,
            self.control,
        )
        det = self.detection
        return {
            "kind": "telemetry",
            "seq": self.seq,
            "t": self.t,
            "stage": self.stage,
            "gimbal": {
                "pan": g.pan,
                "tilt": g.tilt,
                "panRate": g.panRate,
                "tiltRate": g.tiltRate,
            },
            "truth": {
                "azTrue": tr.azTrue,
                "elTrue": tr.elTrue,
                "rangeKm": tr.rangeKm,
            },
            "detection": None
            if det is None
            else {
                "cx": det.cx,
                "cy": det.cy,
                "w": det.w,
                "h": det.h,
                "score": det.score,
                "subpixel": det.subpixel,
            },
            "estimate": {
                "cx": e.cx,
                "cy": e.cy,
                "vx": e.vx,
                "vy": e.vy,
                "sigmaX": e.sigmaX,
                "sigmaY": e.sigmaY,
                "sigmaTheta": e.sigmaTheta,
                "coastFrames": e.coastFrames,
            },
            "error": {
                "px": er.px,
                "py": er.py,
                "norm": er.norm,
                "azErr": er.azErr,
                "elErr": er.elErr,
            },
            "confidence": self.confidence,
            "metrics": {
                "fps": m.fps,
                "acqTimeMs": m.acqTimeMs,
                "lockRetention": m.lockRetention,
                "rmsErrorPx": m.rmsErrorPx,
                "nis": m.nis,
                "computeMs": m.computeMs,
            },
            "disturbance": {
                "cn2": d.cn2,
                "jitterHz": d.jitterHz,
                "jitterUrad": d.jitterUrad,
                "awgnSigma": d.awgnSigma,
                "occlusion": d.occlusion,
            },
            "control": {
                "kp": c.kp,
                "ki": c.ki,
                "kd": c.kd,
                "tunerMode": c.tunerMode,
                "effortPan": c.effortPan,
                "effortTilt": c.effortTilt,
                "criticValue": c.criticValue,
            },
            "scintillation": self.scintillation,
        }


@dataclass
class LogEvent:
    seq: int = 0
    t: float = 0.0
    level: str = "INFO"
    channel: str = "PAT"
    message: str = ""
    kind: str = "log"

    def wire(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunMeta:
    runId: str = ""
    trajectory: str = "UAV_ORBIT"
    detector: str = "CLASSICAL"
    fovDeg: float = 8.0
    width: int = 640
    height: int = 480
    targetHz: float = 60.0
    seed: int = 0
    kind: str = "meta"

    def wire(self) -> dict[str, Any]:
        return asdict(self)


#: Mirrors LOCK_TOLERANCE_PX / COAST_TIMEOUT_FRAMES in types.ts. Duplicated here
#: rather than imported from config so the parity test can compare literals, and
#: `tests/test_schema_parity.py` asserts these equal the config values too.
LOCK_TOLERANCE_PX = 12
COAST_TIMEOUT_FRAMES = 90
