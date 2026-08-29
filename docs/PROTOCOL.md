# SPECTRA GUARD — Transport Protocol

Version `1.0`. This document and `frontend/lib/types.ts` are the contract. The
Python mirror lives in `backend/spectraguard/schema.py` and is validated against
these field names by `backend/tests/test_schema_parity.py`.

## Why two channels

Coarse alignment is only convincing if the jury sees it at video rate. Base64ing
a 640×480 JPEG into JSON inflates it ~33% and forces the browser to parse a
multi-hundred-KB string every 16 ms, which pins the main thread and starves the
canvas. So we split the stream:

| Channel | WS frame type | Payload | Rate |
| --- | --- | --- | --- |
| Telemetry | text | JSON `ServerEvent` | 60 Hz |
| Imagery | binary | raw JPEG bytes | 60 Hz, droppable |

The server emits the `telemetry` event **immediately before** the JPEG for the
same `seq`. The client therefore pairs them positionally and needs no binary
header. If the socket's `bufferedAmount` exceeds the high-water mark, the server
drops **imagery only** and telemetry continues uninterrupted — numbers stay
truthful even when the picture stutters, which is the correct failure mode for a
telemetry console.

## Endpoint

```
ws://127.0.0.1:8000/ws/pat
```

On connect the server sends exactly one `meta` event describing camera
intrinsics and the loop rate, then nothing until a `start` command arrives.

## Sequence

```
client                                server
  |  ── connect ──────────────────────>  |
  |  <──────────── {kind:"meta", ...} ─  |
  |  ── {type:"start", trajectory} ───>  |
  |  <──────── {kind:"log", "PAT", ...}  |
  |  <──────── {kind:"telemetry", seq:1} |
  |  <──────── <binary JPEG seq:1>       |
  |  <──────── {kind:"telemetry", seq:2} |
  |  <──────── <binary JPEG seq:2>       |
  |  ── {type:"set_disturbance", ...} ─> |   (applied on next tick, no ack)
  |  ── {type:"stop"} ────────────────>  |
```

## Commands

All client→server messages are JSON text frames matching `ClientCommand`.
Unknown `type` values are answered with a `log` event at `ERROR` and otherwise
ignored — the engine never dies from a malformed command.

`set_disturbance` and `set_control` take **partial** patches. The slider UI sends
only the axis that moved, at a throttled 20 Hz, so dragging the Cn² slider does
not clobber a concurrent SAC gain update.

## Degraded mode

If the WebSocket cannot be opened, the frontend falls back to
`lib/mock-engine.ts`, an in-browser kinematic simulator that emits the identical
`ServerEvent` shape. The UI cannot tell the difference. This exists so the
dashboard demos on any laptop, projector, or judging table where the Python
process is not running — and so frontend work never blocks on backend work.
The header shows `SIM` instead of `LINK` when this path is active; we do not
hide it, because claiming a live link that isn't there would be dishonest in
front of a jury that will ask.

## Units

Angles are **degrees** on the wire (operators read degrees), radians internally
in Python. Rates are degrees/second. Pixel coordinates are floats with origin at
the top-left of the image, y increasing downward, matching both OpenCV and
Canvas2D so no axis flip is ever needed. `cn2` is m^(−2/3), `jitterUrad` is
microradians RMS, `awgnSigma` is 8-bit DN.
