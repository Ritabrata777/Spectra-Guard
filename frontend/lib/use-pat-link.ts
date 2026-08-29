'use client';

/**
 * SPECTRA GUARD — transport link.
 *
 * Opens the WebSocket to the Python engine and, if that fails, transparently
 * starts the in-browser simulator instead. Both paths feed the same telemetry
 * bus with the same event shape, so no component below this hook knows or cares
 * which one is live.
 *
 * The fallback is not a hack — it is what makes the console demonstrable on a
 * machine with no Python environment, and what stops frontend work from being
 * blocked on backend work. It is surfaced honestly in the header as SIM.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { bus, type LinkMode } from './telemetry-bus';
import { MockEngine } from './mock-engine';
import { useCommandStore } from './store';
import type { ClientCommand, ServerEvent } from './types';

const WS_URL =
  process.env.NEXT_PUBLIC_PAT_WS_URL ?? 'ws://127.0.0.1:8000/ws/pat';

/** How long to wait for the backend before falling back to the simulator. */
const CONNECT_TIMEOUT_MS = 1200;

export interface PatLink {
  linkMode: LinkMode;
  running: boolean;
  start: () => void;
  stop: () => void;
  reset: () => void;
  send: (cmd: ClientCommand) => void;
  exportCsv: () => void;
}

export function usePatLink(): PatLink {
  const [linkMode, setLinkMode] = useState<LinkMode>('OFFLINE');
  const [running, setRunning] = useState(false);

  const socketRef = useRef<WebSocket | null>(null);
  const engineRef = useRef<MockEngine | null>(null);
  const imageSeqRef = useRef(0);
  const setSend = useCommandStore((s) => s.setSend);

  /* --------------------------- image decode path -------------------------- */

  const handleBinary = useCallback(async (data: Blob | ArrayBuffer) => {
    // Tag each decode so a slow one that resolves late cannot overwrite a newer
    // frame. Out-of-order bitmaps look like the video briefly running backwards.
    const mySeq = ++imageSeqRef.current;
    try {
      const blob = data instanceof Blob ? data : new Blob([data], { type: 'image/jpeg' });
      const bitmap = await createImageBitmap(blob);
      if (mySeq !== imageSeqRef.current) {
        bitmap.close();
        return;
      }
      bus.setImage(bitmap, bus.latest?.seq ?? mySeq);
    } catch {
      // A corrupt JPEG must never take the console down; the numbers are still
      // arriving on the text channel and remain the authoritative record.
    }
  }, []);

  /* ------------------------------ connection ------------------------------ */

  useEffect(() => {
    let disposed = false;
    let fallbackTimer: ReturnType<typeof setTimeout> | null = null;

    const startSim = () => {
      if (disposed || engineRef.current) return;
      const store = useCommandStore.getState();
      engineRef.current = new MockEngine({
        getDisturbance: () => useCommandStore.getState().disturbance,
        getTrajectory: () => useCommandStore.getState().trajectory,
        getGains: () => {
          const s = useCommandStore.getState();
          return { kp: s.kp, ki: s.ki, kd: s.kd, tunerMode: s.tunerMode };
        },
      });
      void store;
      bus.setLinkMode('SIM');
      setLinkMode('SIM');
      bus.pushLog({
        kind: 'log',
        seq: 0,
        t: 0,
        level: 'WARN',
        channel: 'LINK',
        message: `No engine at ${WS_URL} — running the in-browser simulator (SIM)`,
      });
    };

    let ws: WebSocket;
    try {
      ws = new WebSocket(WS_URL);
    } catch {
      startSim();
      return () => {
        disposed = true;
      };
    }

    ws.binaryType = 'blob';
    socketRef.current = ws;

    fallbackTimer = setTimeout(() => {
      if (ws.readyState !== WebSocket.OPEN) {
        try {
          ws.close();
        } catch {
          /* already closing */
        }
        startSim();
      }
    }, CONNECT_TIMEOUT_MS);

    ws.onopen = () => {
      if (fallbackTimer) clearTimeout(fallbackTimer);
      if (disposed) return;
      bus.setLinkMode('LINK');
      setLinkMode('LINK');
      bus.pushLog({
        kind: 'log',
        seq: 0,
        t: 0,
        level: 'OK',
        channel: 'LINK',
        message: `Engine link established — ${WS_URL}`,
      });
    };

    ws.onmessage = (ev) => {
      if (typeof ev.data === 'string') {
        try {
          const msg = JSON.parse(ev.data) as ServerEvent;
          if (msg.kind === 'telemetry') bus.pushFrame(msg);
          else if (msg.kind === 'log') bus.pushLog(msg);
          else if (msg.kind === 'meta') bus.setMeta(msg);
        } catch {
          /* Malformed JSON is the backend's bug to fix, not a reason to die. */
        }
      } else {
        void handleBinary(ev.data as Blob);
      }
    };

    ws.onerror = () => {
      if (fallbackTimer) clearTimeout(fallbackTimer);
      startSim();
    };

    ws.onclose = () => {
      if (disposed) return;
      if (fallbackTimer) clearTimeout(fallbackTimer);
      socketRef.current = null;
      // If we had a live link and lost it, say so plainly and keep the console
      // usable on the simulator rather than freezing on a stale last frame.
      if (!engineRef.current) {
        bus.pushLog({
          kind: 'log',
          seq: 0,
          t: 0,
          level: 'ERROR',
          channel: 'LINK',
          message: 'Engine link closed — falling back to simulator',
        });
        startSim();
      }
    };

    return () => {
      disposed = true;
      if (fallbackTimer) clearTimeout(fallbackTimer);
      engineRef.current?.stop();
      engineRef.current = null;
      try {
        ws.close();
      } catch {
        /* nothing to do */
      }
    };
  }, [handleBinary]);

  /* -------------------------------- commands ------------------------------ */

  const send = useCallback((cmd: ClientCommand) => {
    const ws = socketRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(cmd));
    }
    // Simulator commands that change loop state are applied directly; parameter
    // patches are read live from the store by the engine each tick, so they need
    // no explicit plumbing here.
    const engine = engineRef.current;
    if (engine) {
      if (cmd.type === 'start') engine.start();
      else if (cmd.type === 'stop') engine.stop();
      else if (cmd.type === 'reset') engine.reset();
    }
  }, []);

  useEffect(() => {
    setSend(send);
    return () => setSend(null);
  }, [send, setSend]);

  const start = useCallback(() => {
    const { trajectory } = useCommandStore.getState();
    send({ type: 'start', trajectory });
    setRunning(true);
  }, [send]);

  const stop = useCallback(() => {
    send({ type: 'stop' });
    setRunning(false);
  }, [send]);

  const reset = useCallback(() => {
    send({ type: 'reset' });
    setRunning(false);
  }, [send]);

  /**
   * Export the performance log. Built from the bus's own ring buffers, so the
   * CSV is provably the same data the operator was watching — not a re-run.
   */
  const exportCsv = useCallback(() => {
    const csv = bus.toCsv();
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const runId = bus.meta?.runId ?? 'RUN';
    a.href = url;
    a.download = `spectraguard-${runId}-performance.csv`;
    a.click();
    URL.revokeObjectURL(url);
    bus.pushLog({
      kind: 'log',
      seq: bus.latest?.seq ?? 0,
      t: bus.latest?.t ?? 0,
      level: 'OK',
      channel: 'LOG',
      message: `Performance log exported — ${bus.history.t.length} samples`,
    });
  }, []);

  return { linkMode, running, start, stop, reset, send, exportCsv };
}

/**
 * Subscribe to the bus's low-frequency state from React.
 *
 * Deliberately NOT useSyncExternalStore over the whole frame: that would tie a
 * render to every telemetry tick, which is precisely the thing telemetry-bus.ts
 * is built to avoid. This hook only ever sees throttled slow state.
 */
export function useSlowTelemetry() {
  const [state, setState] = useState(bus.getSlowState);
  useEffect(() => bus.subscribeSlow(setState), []);
  return state;
}
