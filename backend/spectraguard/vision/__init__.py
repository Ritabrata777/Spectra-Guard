"""
SPECTRA GUARD vision: beacon detection and sub-pixel centroiding.

Importing this package pulls in only numpy and OpenCV. The YOLOv11 backend's
``ultralytics``/torch import happens inside ``YoloV11BeaconDetector.__init__``,
so the neural path costs nothing until it is actually requested.
"""

from .centroid import CentroidResult, annulus_background, refine
from .detector import (
    BeaconDetector,
    ClassicalBeaconDetector,
    Detection,
    DetectionPrior,
    DetectorUnavailable,
    YoloV11BeaconDetector,
    build_detector,
)

__all__ = [
    "CentroidResult",
    "annulus_background",
    "refine",
    "BeaconDetector",
    "ClassicalBeaconDetector",
    "Detection",
    "DetectionPrior",
    "DetectorUnavailable",
    "YoloV11BeaconDetector",
    "build_detector",
]
