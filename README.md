# Spectra Guard

Spectra Guard is a closed-loop coarse pointing, acquisition, and tracking (PAT) simulator for a mobile free-space optical communications terminal. The project combines a Python physics/control engine with a Next.js dashboard that visualizes telemetry and live camera imagery over a WebSocket stream.

## Overview

- Backend: Python + FastAPI + OpenCV + NumPy
- Frontend: Next.js + React + TypeScript
- Protocol: JSON telemetry + JPEG imagery over WebSocket
- Goal: model optical turbulence, estimation, control, and gimbal behavior in a real-time PAT loop

## Repository structure

```text
Spectra-Guard/
├── backend/
│   ├── spectraguard/
│   ├── tests/
│   └── requirements.txt
├── frontend/
│   ├── app/
│   ├── components/
│   ├── lib/
│   ├── package.json
│   └── package-lock.json
├── docs/
│   ├── ARCHITECTURE.md
│   └── PROTOCOL.md
├── .gitignore
└── README.md
```

## Getting started

### 1) Set up the backend

```bash
cd backend
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
```

Start the backend server:

```bash
cd backend
python -m uvicorn spectraguard.server:app --host 127.0.0.1 --port 8000 --reload
```

The backend exposes:

- http://127.0.0.1:8000/
- http://127.0.0.1:8000/health
- ws://127.0.0.1:8000/ws/pat

### 2) Set up the frontend

```bash
cd frontend
npm install
npm run dev
```

Then open:

- http://localhost:3000

## Project behavior

The system simulates a PAT loop with:

- optical propagation and scene generation
- atmospheric turbulence and vibration modeling
- detector/image-plane measurement generation
- an EKF-based estimator
- PID-based control and gimbal motion
- live telemetry and imagery streaming to the browser

The backend owns the physical logic and the frontend renders the dashboard contract. The protocol is documented in [docs/PROTOCOL.md](docs/PROTOCOL.md), and the architecture is described in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Testing

Run backend tests:

```bash
cd backend
pytest
```

The project also includes lightweight validation scripts in the backend test suite for the protocol and physics logic.

## Notes

- Backend telemetry uses SI units internally and converts to degrees/pixels on the wire.
- The dashboard supports a simulated degraded mode if the live WebSocket is unavailable.
- The project is designed for a real-time PAT demo and is optimized for browser-side telemetry visualization.

## License

This project is currently unlicensed unless a repository-level license file is added later.
