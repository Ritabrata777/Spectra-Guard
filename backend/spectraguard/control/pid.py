"""Two independent filtered PID rate controllers with back-calculation clamp."""
from __future__ import annotations
import math
from dataclasses import replace
from ..config import PidConfig

class RatePID:
    def __init__(self, cfg: PidConfig) -> None:
        self.cfg = cfg; self.integral = [0.,0.]; self.derivative = [0.,0.]; self.prev = [0.,0.]
    def reset(self) -> None: self.integral=[0.,0.]; self.derivative=[0.,0.]; self.prev=[0.,0.]
    def set_gains(self, kp: float|None=None, ki: float|None=None, kd: float|None=None) -> None:
        self.cfg = replace(self.cfg, kp=self.cfg.kp if kp is None else kp, ki=self.cfg.ki if ki is None else ki, kd=self.cfg.kd if kd is None else kd)
    def step(self, ex_px: float, ey_px: float, dt: float) -> tuple[float,float]:
        out=[]; alpha = math.exp(-2*math.pi*self.cfg.d_cutoff_hz*dt)
        for i,e_px in enumerate((ex_px, ey_px)):
            e=e_px/self.cfg.error_scale; raw=(e-self.prev[i])/max(dt,1e-6)
            self.derivative[i]=alpha*self.derivative[i]+(1-alpha)*raw
            candidate=max(-self.cfg.integ_limit,min(self.cfg.integ_limit,self.integral[i]+e*dt))
            u=self.cfg.kp*e+self.cfg.ki*candidate+self.cfg.kd*self.derivative[i]
            clipped=max(-self.cfg.out_limit_rad_s,min(self.cfg.out_limit_rad_s,u))
            if clipped == u or (u*e < 0): self.integral[i]=candidate
            self.prev[i]=e; out.append(clipped)
        return out[0], -out[1]  # +pixel y means target below, so tilt downward
