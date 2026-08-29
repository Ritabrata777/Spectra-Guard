"""FastAPI WebSocket adapter; all PAT behaviour lives in :mod:`engine`."""
from __future__ import annotations
import asyncio, json
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from .engine import PatEngine

app=FastAPI(title="Spectra Guard PAT")
@app.get("/")
def root() -> dict:
    """Human-friendly landing response for a browser opened on the API port."""
    return {
        "service": "Spectra Guard PAT backend",
        "status": "running",
        "dashboard": "http://localhost:3000",
        "health": "/health",
        "websocket": "/ws/pat",
    }

@app.get("/health")
def health() -> dict:
    return {"ok": True}

@app.websocket("/ws/pat")
async def pat_socket(ws: WebSocket) -> None:
    await ws.accept()
    engine = PatEngine()
    await ws.send_json(engine.meta().wire())

    async def receive() -> None:
        while True:
            try:
                raw = await ws.receive_text()
                cmd = json.loads(raw)
                typ = cmd.get("type")
                if typ == "start":
                    traj = cmd.get("trajectory")
                    if traj:
                        try:
                            engine.set_trajectory(traj)
                        except Exception as e:
                            await ws.send_json(engine._log("WARN", "WS", f"Invalid trajectory '{traj}': {e}").wire())
                    await ws.send_json(engine.start().wire())
                elif typ == "stop":
                    await ws.send_json(engine.stop().wire())
                elif typ == "reset":
                    await ws.send_json(engine.reset().wire())
                elif typ == "set_disturbance":
                    engine.patch_disturbance(cmd.get("patch", {}))
                elif typ == "set_control":
                    engine.patch_control(cmd.get("patch", {}))
                elif typ == "set_trajectory":
                    traj = cmd.get("trajectory", engine.trajectory.kind)
                    try:
                        engine.set_trajectory(traj)
                    except Exception as e:
                        await ws.send_json(engine._log("WARN", "WS", f"Invalid trajectory '{traj}': {e}").wire())
                else:
                    await ws.send_json(engine._log("ERROR", "WS", f"unknown command: {typ}").wire())
            except WebSocketDisconnect:
                break
            except Exception as e:
                try:
                    await ws.send_json(engine._log("ERROR", "WS", f"command processing error: {e}").wire())
                except Exception:
                    break

    task = asyncio.create_task(receive())
    try:
        while True:
            if engine.running:
                frame, image, events = engine.tick()
                # Discrete logs are emitted before the telemetry/JPEG pair: the
                # protocol guarantees those two frames remain adjacent.
                for event in events:
                    await ws.send_json(event.wire())
                await ws.send_json(frame.wire())
                ok, jpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    await ws.send_bytes(jpg.tobytes())
            await asyncio.sleep(engine.cfg.dt)
    except WebSocketDisconnect:
        pass
    finally:
        task.cancel()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("spectraguard.server:app", host="127.0.0.1", port=8000, reload=False)

