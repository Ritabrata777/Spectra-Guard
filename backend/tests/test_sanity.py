import math
from spectraguard.config import DEFAULT_CONFIG
from spectraguard.estimation import BearingEKF
from spectraguard.physics.turbulence import fried_parameter, rytov_variance
from spectraguard.sim.camera import PinholeCamera
from spectraguard.vision.detector import Detection

def test_projection_round_trip():
    cam=PinholeCamera(DEFAULT_CONFIG.camera); az,el=.01,.02
    u,v=cam.project(az,el,-.005,.003)
    got=cam.unproject(u,v,-.005,.003)
    assert got == pytest.approx((az,el), abs=1e-12)

def test_turbulence_strength_ordering():
    c=DEFAULT_CONFIG.camera; t=DEFAULT_CONFIG.turbulence
    assert fried_parameter(1e-14,c.wavenumber,t.path_length_m) < fried_parameter(1e-15,c.wavenumber,t.path_length_m)
    assert rytov_variance(1e-14,c.wavenumber,t.path_length_m) > rytov_variance(1e-15,c.wavenumber,t.path_length_m)

def test_ekf_rejects_remote_measurement():
    cam=PinholeCamera(DEFAULT_CONFIG.camera); ekf=BearingEKF(DEFAULT_CONFIG.ekf,cam)
    ekf.seed(Detection(cam.cx,cam.cy,3,3,1,True,1,1,1,1),0,0); ekf.predict(DEFAULT_CONFIG.dt)
    assert not ekf.update(Detection(0,0,3,3,1,True,1,1,1,1),0,0)

import pytest
