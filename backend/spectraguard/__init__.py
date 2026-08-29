"""SPECTRA GUARD — AI-based virtual camera tracking for mobile FSOC terminals.

Smart India Hackathon · ISRO problem statement · Team Incu3bit.

LAYOUT, and why it is shaped this way:

    config.py       every constant in the system, units declared once
    schema.py       the wire contract, mirrored from frontend/lib/types.ts

    physics/        the truth model — turbulence, platform vibration, sensor noise
    sim/            target trajectories, pinhole camera, frame synthesis
    vision/         detection: YOLOv11 when weights exist, classical otherwise
    estimation/     the bearing-space EKF
    control/        PID plus the adaptive gain tuner
    telemetry/      metrics accumulation and the log book

    engine.py       the PAT state machine that wires the above into one loop
    server.py       a thin FastAPI/WebSocket shell around engine.py
    cli.py          the same engine, headless, straight to CSV

THE ONE STRUCTURAL RULE. `engine.py` imports FastAPI nowhere, and `server.py`
contains no algorithm. Everything that can be wrong about the physics, the filter
or the controller is therefore reachable from a plain `python -m tests` with no
web stack installed. This is not architectural purity for its own sake — it is
what makes the numbers in the technical report checkable by someone who has only
numpy, and it is what let this codebase be developed in an environment where
`pip install fastapi` was blocked.

A SECOND RULE, about honesty. Nothing in this package ever silently substitutes
a fallback. If YOLOv11 weights are absent the detector backend reports
`CLASSICAL` on the wire and the UI header says so; if torch is missing the tuner
reports CEM rather than SAC. A demo that quietly lies about which code path
produced a result is worth less than one that admits its configuration.
"""

__all__ = ["__version__"]

__version__ = "1.0.0"
