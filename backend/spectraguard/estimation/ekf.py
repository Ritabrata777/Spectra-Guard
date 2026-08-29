"""Six-state, nearly-constant-acceleration EKF in inertial bearing space."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import CHI2_GATE_2DOF_99, EkfConfig
from ..sim.camera import PinholeCamera
from ..vision.detector import Detection


@dataclass(frozen=True)
class EkfSnapshot:
    cx: float; cy: float; vx: float; vy: float
    sigma_x: float; sigma_y: float; sigma_theta: float
    coast_frames: int; nis: float; accepted: bool


class BearingEKF:
    """Tracks [az, el, az_rate, el_rate, az_acc, el_acc] in SI radians."""
    def __init__(self, cfg: EkfConfig, camera: PinholeCamera) -> None:
        self.cfg, self.camera = cfg, camera
        self.x = np.zeros(6); self.P = np.eye(6)
        self.initialized = False; self.coast_frames = 0; self.last_nis = 0.0
        self.nis_history: deque[float] = deque(maxlen=cfg.nis_window)

    def reset(self) -> None:
        self.initialized = False; self.coast_frames = 0; self.last_nis = 0.0
        self.nis_history.clear()

    def seed(self, det: Detection, pan: float, tilt: float) -> None:
        az, el = self.camera.unproject(det.u, det.v, pan, tilt)
        self.x[:] = (az, el, 0, 0, 0, 0)
        p, v, a = self.cfg.init_sigma_pos_rad, self.cfg.init_sigma_vel_rad_s, self.cfg.init_sigma_acc_rad_s2
        self.P = np.diag([p*p, p*p, v*v, v*v, a*a, a*a])
        self.initialized = True; self.coast_frames = 0

    def predict(self, dt: float) -> None:
        if not self.initialized: return
        F = np.eye(6)
        for i in (0, 1):
            F[i, i+2] = dt; F[i, i+4] = .5*dt*dt; F[i+2, i+4] = dt
        q = self.cfg.sigma_jerk**2
        q1 = np.array([[dt**5/20, dt**4/8, dt**3/6], [dt**4/8, dt**3/3, dt**2/2], [dt**3/6, dt**2/2, dt]]) * q
        Q = np.zeros((6, 6)); Q[np.ix_((0,2,4),(0,2,4))] = q1; Q[np.ix_((1,3,5),(1,3,5))] = q1
        self.x = F @ self.x; self.P = F @ self.P @ F.T + Q
        self.P = .5 * (self.P + self.P.T)

    def prior(self, pan: float, tilt: float) -> Optional[tuple[float,float,float,float,float]]:
        if not self.initialized: return None
        u, v = self.camera.project(self.x[0], self.x[1], pan, tilt)
        ju, jv = self.camera.jacobian(self.x[0], self.x[1], pan, tilt)
        su, sv = math.sqrt(max(1e-9, ju*ju*self.P[0,0])), math.sqrt(max(1e-9, jv*jv*self.P[1,1]))
        return u, v, su, sv, max(self.cfg.reacquire_min_px, self.cfg.reacquire_sigma * max(su, sv))

    def update(self, det: Optional[Detection], pan: float, tilt: float) -> bool:
        if det is None:
            if self.initialized: self.coast_frames += 1
            return False
        if not self.initialized:
            self.seed(det, pan, tilt); return True
        u, v = self.camera.project(self.x[0], self.x[1], pan, tilt)
        ju, jv = self.camera.jacobian(self.x[0], self.x[1], pan, tilt)
        H = np.zeros((2,6)); H[0,0] = ju; H[1,1] = jv
        sigma = self.cfg.sigma_pix_base * (1 + self.cfg.score_inflation * (1 - max(0., min(1., det.score))))
        R = np.eye(2) * sigma*sigma
        innov = np.array([det.u-u, det.v-v]); S = H @ self.P @ H.T + R
        try: nis = float(innov @ np.linalg.solve(S, innov))
        except np.linalg.LinAlgError: nis = float('inf')
        self.last_nis = nis; self.nis_history.append(nis)
        if not math.isfinite(nis) or nis > CHI2_GATE_2DOF_99:
            self.coast_frames += 1; return False
        K = np.linalg.solve(S, H @ self.P).T
        self.x += K @ innov
        I = np.eye(6); self.P = (I-K@H) @ self.P @ (I-K@H).T + K@R@K.T
        self.P = .5*(self.P+self.P.T); self.coast_frames = 0
        return True

    def snapshot(self, pan: float, tilt: float) -> EkfSnapshot:
        if not self.initialized: return EkfSnapshot(0,0,0,0,0,0,0,0,0,False)
        u,v = self.camera.project(self.x[0],self.x[1],pan,tilt); ju,jv = self.camera.jacobian(self.x[0],self.x[1],pan,tilt)
        sx,sy = abs(ju)*math.sqrt(max(0,self.P[0,0])), abs(jv)*math.sqrt(max(0,self.P[1,1]))
        cov = np.array([[ju*ju*self.P[0,0], ju*jv*self.P[0,1]],[ju*jv*self.P[1,0], jv*jv*self.P[1,1]]])
        theta = .5*math.atan2(2*cov[0,1], cov[0,0]-cov[1,1])
        return EkfSnapshot(u,v,ju*self.x[2],jv*self.x[3],sx,sy,theta,self.coast_frames,self.last_nis,True)

    @property
    def mean_nis(self) -> float:
        return float(np.mean(self.nis_history)) if self.nis_history else 0.0
