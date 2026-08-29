# Spectra Guard architecture

Spectra Guard is a closed-loop coarse pointing, acquisition and tracking (PAT)
simulator for a mobile free-space optical communications terminal. The browser
only renders the WebSocket contract; the Python engine owns every physical,
estimation and control decision.

```text
trajectory + atmosphere + vibration --> camera scene --> detector --> EKF --> PID --> gimbal
          ^                                                               |             |
          +--------------------------- achieved boresight ----------------+-------------+
```

## Coordinate and transport contract

The runtime uses SI units and radians. The wire uses degrees for bearings and
rates, pixels for focal-plane coordinates, and camelCase JSON fields defined in
`frontend/lib/types.ts` and mirrored by `backend/spectraguard/schema.py`.
`docs/PROTOCOL.md` defines the two-frame WebSocket stream: telemetry JSON is
sent immediately before its raw JPEG image.

## Optical channel and image formation

For focal length in pixels, with horizontal FOV \(\alpha\),

\[
f_{px}=\frac{W/2}{\tan(\alpha/2)},\quad
u=c_x+f_{px}\tan(az-pan),\quad
v=c_y-f_{px}\tan(el-tilt).
\]

Atmosphere uses a von Kármán phase screen for correlated angle-of-arrival
tilt, frozen flow for time evolution, and closed-form turbulence statistics:

\[
r_0=(0.423 k^2 C_n^2 L)^{-3/5},\qquad
\sigma_R^2=1.23 C_n^2 k^{7/6}L^{11/6}.
\]

The scene broadens the PSF by \(\max(1,D/r_0)^{5/6}\), conserving flux, and
draws aperture-averaged log-normal scintillation. Platform vibration is a
calibrated narrowband biquad, deliberately separate from optical turbulence.

## Estimation and control

The EKF state is
\(x=[az,el,\dot{az},\dot{el},\ddot{az},\ddot{el}]^T\). It follows a
nearly-constant-acceleration process with white jerk process noise. The
nonlinear measurement is the pinhole projection above, with Jacobian
\(H_{00}=f\sec^2(az-pan)\), \(H_{11}=-f\sec^2(el-tilt)\). Measurements are
accepted only when normalized innovation squared is below the 99% two-degree
of-freedom gate, \(\mathrm{NIS}\le9.2103\).

The PID controls image error as a gimbal rate command. Its derivative term is
low-pass filtered, its integral is bounded, and its output is limited to the
physical slew rate. The plant is a first-order rate servo:

\[
\tau\dot\omega=-\omega+K u,\qquad \dot\theta=\omega.
\]

The state machine is `IDLE → SEARCH → ACQUIRE → TRACK`, with `COAST` for
temporary dropouts and `FAULT → SEARCH` after coast timeout. This makes loss
of optical evidence visible rather than silently continuing to report lock.

## Jury checks

Run `pytest` from `backend`, then start the server with
`uvicorn spectraguard.server:app --reload`. The sanity tests verify projection
round-trip, turbulence ordering, EKF gate behavior, and schema defaults.
