"""Deterministic PAT runtime: simulation, detection, EKF and gimbal control."""
from __future__ import annotations
import math, time, uuid
from dataclasses import replace
from typing import Optional
import numpy as np
from .config import AppConfig, DEFAULT_CONFIG, Disturbance
from .control import RatePID
from .estimation import BearingEKF
from .physics.jitter import PlatformJitter
from .physics.noise import SensorNoise
from .physics.turbulence import TurbulenceField
from .schema import (ControlState, Detection as WireDetection, Disturbance as WireDisturbance,
 GimbalState, EkfEstimate, LogEvent, PerformanceMetrics, RunMeta, TargetTruth, TelemetryFrame, TrackingError)
from .sim.camera import Gimbal, PinholeCamera
from .sim.scene import SceneRenderer
from .sim.trajectory import Trajectory
from .vision.detector import DetectionPrior, build_detector

class PatEngine:
    def __init__(self, config: AppConfig = DEFAULT_CONFIG, seed: Optional[int] = None) -> None:
        self.cfg=config; self.seed=config.seed if seed is None else seed; self.run_id=uuid.uuid4().hex[:12]
        self._build(); self.running=False; self.stage="IDLE"; self.seq=0; self.t=0.; self.logs:list[LogEvent]=[]
        self._search_started=None; self._acq_time=None; self._track_frames=0; self._post_acq_frames=0; self._rms_sq=0.
        self._lock_count=self._miss_count=self._reacquire_count=self._fault_count=0; self._last_cmd=(0.,0.)

    def _build(self) -> None:
        rng=np.random.default_rng(self.seed); parts=[np.random.default_rng(rng.integers(2**32)) for _ in range(6)]
        self.cam=PinholeCamera(self.cfg.camera); self.gimbal=Gimbal(self.cfg.camera, -self.cfg.initial_offset_rad, 0.)
        self.trajectory=Trajectory(self.cfg.trajectory, rng=parts[0]); self.noise=SensorNoise(self.cfg.noise, parts[1])
        self.scene=SceneRenderer(self.cfg.scene,self.cfg.camera,self.noise,parts[2]); self.turb=TurbulenceField(self.cfg.turbulence,self.cfg.camera,self.cfg.disturbance,parts[3])
        self.jitter=PlatformJitter(self.cfg.jitter,self.cfg.pat.target_hz,self.cfg.disturbance.jitter_hz,self.cfg.disturbance.jitter_urad,parts[4])
        self.detector,self.detector_note=build_detector(self.cfg.detector); self.ekf=BearingEKF(self.cfg.ekf,self.cam); self.pid=RatePID(self.cfg.pid); self.disturbance=self.cfg.disturbance

    def meta(self) -> RunMeta:
        return RunMeta(runId=self.run_id,trajectory=self.trajectory.kind,detector=self.detector.name,fovDeg=self.cfg.camera.fov_deg,width=self.cfg.camera.width,height=self.cfg.camera.height,targetHz=self.cfg.pat.target_hz,seed=self.seed)
    def _log(self, level:str, channel:str, message:str) -> LogEvent:
        ev=LogEvent(seq=self.seq,t=self.t,level=level,channel=channel,message=message); self.logs.append(ev); return ev
    def start(self) -> LogEvent:
        self.running=True; self.stage="SEARCH"; self._search_started=self.t; self._log("INFO","OPT",self.detector_note); return self._log("INFO","PAT","search pattern armed")
    def stop(self) -> LogEvent:
        self.running=False; self.stage="IDLE"; self._last_cmd=(0.,0.); return self._log("INFO","PAT","hold command received")
    def reset(self) -> LogEvent:
        self._build(); self.stage="IDLE"; self.seq=0; self.t=0.; self.running=False; return self._log("OK","PAT","run reset")
    def patch_disturbance(self, patch: dict) -> None:
        snake={"jitterHz":"jitter_hz","jitterUrad":"jitter_urad","awgnSigma":"awgn_sigma"}
        self.disturbance=self.disturbance.patched({snake.get(k,k):v for k,v in patch.items()}); self.turb.update(self.disturbance); self.jitter.set_params(self.disturbance.jitter_hz,self.disturbance.jitter_urad)
    def patch_control(self, patch: dict) -> None:
        self.pid.set_gains(patch.get("kp"),patch.get("ki"),patch.get("kd"))
    def set_trajectory(self, kind: str) -> None:
        self.trajectory=Trajectory(self.cfg.trajectory, kind=kind, rng=np.random.default_rng(self.seed)); self._log("INFO","PAT",f"trajectory={kind}")
    def _search_command(self) -> tuple[float, float]:
        """Expanding circular scan; bounded by the configured search cone."""
        age=max(0., self.t-(self._search_started or self.t)); p=self.cfg.pat
        radius=min(p.search_cone_rad, p.search_pitch_rad * age / (2*math.pi))
        phase=p.search_speed_rad_s * age / max(radius, p.search_pitch_rad)
        return p.search_speed_rad_s*math.cos(phase), p.search_speed_rad_s*math.sin(phase)
    def tick(self) -> tuple[TelemetryFrame, np.ndarray, list[LogEvent]]:
        started=time.perf_counter(); dt=self.cfg.dt; emitted=[]
        if not self.running:
            return self._frame(None, None, 1., 0., time.perf_counter()-started), np.zeros((self.cfg.camera.height,self.cfg.camera.width,3),np.uint8), emitted
        self.seq+=1; self.t+=dt; truth=self.trajectory.state(self.t); turb=self.turb.sample(dt); jit=self.jitter.step()
        g=self.gimbal.step(*self._last_cmd,dt,jit.pan_rad,jit.tilt_rad)
        uv=self.cam.project(truth.az,truth.el,g.pan_achieved,g.tilt_achieved)
        render=self.scene.render(self.t,uv,turb,self.disturbance,self.cam)
        self.ekf.predict(dt)
        prior=self.ekf.prior(g.pan_reported,g.tilt_reported)
        prior_obj=None if prior is None else DetectionPrior(*prior)
        detection=self.detector.detect(render.frame,prior_obj,render.psf_sigma_px)
        accepted=self.ekf.update(detection,g.pan_reported,g.tilt_reported)
        snap=self.ekf.snapshot(g.pan_reported,g.tilt_reported)
        if detection and accepted:
            self._miss_count=0; self._lock_count+=1; self._reacquire_count+=1
            if self.stage=="SEARCH": self.stage="ACQUIRE"; emitted.append(self._log("INFO","PAT","beacon candidate acquired"))
            if self.stage=="ACQUIRE" and self._lock_count>=self.cfg.pat.lock_frames:
                self.stage="TRACK"; self._acq_time=(self.t-(self._search_started or self.t))*1000; emitted.append(self._log("OK","PAT","closed-loop lock"))
            if self.stage=="COAST" and self._reacquire_count>=self.cfg.pat.reacquire_frames:
                self.stage="TRACK"; emitted.append(self._log("OK","EKF","reacquired measurement"))
        else:
            self._lock_count=0; self._reacquire_count=0; self._miss_count+=1
            if self.stage in ("ACQUIRE","TRACK") and self._miss_count>=self.cfg.pat.miss_frames_to_coast:
                self.stage="COAST"; emitted.append(self._log("WARN","PAT","measurement lost; coasting"))
            if self.stage=="COAST" and snap.coast_frames>=self.cfg.pat.coast_timeout_frames:
                self.stage="FAULT"; self._fault_count=self.cfg.pat.fault_hold_frames; emitted.append(self._log("ERROR","PAT","coast timeout; restarting search"))
        if self.stage=="FAULT":
            self._fault_count-=1
            if self._fault_count<=0: self.stage="SEARCH"; self.ekf.reset()
        ex = (snap.cx - self.cfg.camera.cx) if snap.accepted else 0.0
        ey = (snap.cy - self.cfg.camera.cy) if snap.accepted else 0.0
        max_err = float(self.cfg.camera.width * 2)
        ex = max(-max_err, min(max_err, ex))
        ey = max(-max_err, min(max_err, ey))
        if self.stage in ("ACQUIRE", "TRACK", "COAST"):
            self._last_cmd = self.pid.step(ex, ey, dt)

        elif self.stage=="SEARCH":
            self._last_cmd=self._search_command()
        else:
            self._last_cmd=(0.,0.)
        return self._frame(detection,render,render.attenuation,truth.range_m,time.perf_counter()-started,g,snap), render.frame, emitted
    def _frame(self, det, render, scint, rng_m, elapsed, g=None, snap=None) -> TelemetryFrame:
        if g is None: g=self.gimbal.step(0,0,0)
        if snap is None: snap=self.ekf.snapshot(g.pan_reported,g.tilt_reported)
        tr=self.trajectory.state(self.t); px=(snap.cx-self.cfg.camera.cx) if snap.accepted else 0.; py=(snap.cy-self.cfg.camera.cy) if snap.accepted else 0.; norm=math.hypot(px,py)
        if self.stage=="TRACK": self._track_frames+=1
        if self._acq_time is not None: self._post_acq_frames+=1; self._rms_sq+=norm*norm
        conf=(det.score if det and snap.accepted else 0.) * math.exp(-snap.coast_frames/30)
        return TelemetryFrame(seq=self.seq,t=self.t,stage=self.stage,
          gimbal=GimbalState(math.degrees(g.pan_reported),math.degrees(g.tilt_reported),math.degrees(g.pan_rate),math.degrees(g.tilt_rate)),
          truth=TargetTruth(math.degrees(tr.az),math.degrees(tr.el),tr.range_m/1000),
          detection=None if det is None else WireDetection(det.u,det.v,det.w,det.h,det.score,det.subpixel),
          estimate=EkfEstimate(snap.cx,snap.cy,snap.vx,snap.vy,snap.sigma_x,snap.sigma_y,snap.sigma_theta,snap.coast_frames),
          error=TrackingError(px,py,norm,math.degrees(px/self.cam.f_px),math.degrees(-py/self.cam.f_px)),confidence=conf,
          metrics=PerformanceMetrics(self.cfg.pat.target_hz,self._acq_time,self._track_frames/max(1,self._post_acq_frames),math.sqrt(self._rms_sq/max(1,self._post_acq_frames)),self.ekf.mean_nis,elapsed*1000),
          disturbance=WireDisturbance(self.disturbance.cn2,self.disturbance.jitter_hz,self.disturbance.jitter_urad,self.disturbance.awgn_sigma,self.disturbance.occlusion),
          control=ControlState(self.pid.cfg.kp,self.pid.cfg.ki,self.pid.cfg.kd,self.cfg.tuner.mode,math.degrees(self._last_cmd[0]),math.degrees(self._last_cmd[1]),None),scintillation=scint)
