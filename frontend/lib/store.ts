/**
 * Operator-owned command state.
 *
 * The split versus `telemetry-bus.ts` is strict and worth stating: the bus holds
 * what the SYSTEM reports (60 Hz, never in React state); this store holds what
 * the OPERATOR commands (slider positions, selected trajectory, gain overrides).
 * Command state changes at human speed, so React state is exactly right for it.
 *
 * Keeping them separate also prevents a subtle class of bug: if slider values
 * were mirrored from incoming telemetry, a slow ack from the backend would yank
 * the knob out from under the operator's finger mid-drag.
 */

import { create } from 'zustand';
import {
  DEFAULT_CONTROL,
  DEFAULT_DISTURBANCE,
  type ClientCommand,
  type Disturbance,
  type TrajectoryKind,
  type TunerMode,
} from './types';

export interface CommandState {
  trajectory: TrajectoryKind;
  disturbance: Disturbance;
  kp: number;
  ki: number;
  kd: number;
  tunerMode: TunerMode;
  /** Operator has expanded the disturbance pill into its full slider matrix. */
  injectorOpen: boolean;
  /** Terminal expanded to full height. */
  terminalExpanded: boolean;

  /** Wired up by the link hook; null until a transport exists. */
  send: ((cmd: ClientCommand) => void) | null;

  setTrajectory: (t: TrajectoryKind) => void;
  setDisturbance: (patch: Partial<Disturbance>) => void;
  setGains: (patch: Partial<Pick<CommandState, 'kp' | 'ki' | 'kd' | 'tunerMode'>>) => void;
  setInjectorOpen: (open: boolean) => void;
  setTerminalExpanded: (open: boolean) => void;
  setSend: (fn: ((cmd: ClientCommand) => void) | null) => void;
  resetDisturbance: () => void;
}

export const useCommandStore = create<CommandState>((set, get) => ({
  trajectory: 'UAV_ORBIT',
  disturbance: { ...DEFAULT_DISTURBANCE },
  ...DEFAULT_CONTROL,
  injectorOpen: false,
  terminalExpanded: false,
  send: null,

  setTrajectory: (trajectory) => {
    set({ trajectory });
    get().send?.({ type: 'set_trajectory', trajectory });
  },

  setDisturbance: (patch) => {
    set((s) => ({ disturbance: { ...s.disturbance, ...patch } }));
    // Send only the axis that moved. A full-object send would clobber a
    // concurrent update from another control mid-drag.
    get().send?.({ type: 'set_disturbance', patch });
  },

  setGains: (patch) => {
    set(patch as Partial<CommandState>);
    get().send?.({ type: 'set_control', patch });
  },

  setInjectorOpen: (injectorOpen) => set({ injectorOpen }),
  setTerminalExpanded: (terminalExpanded) => set({ terminalExpanded }),
  setSend: (send) => set({ send }),

  resetDisturbance: () => {
    const disturbance = { ...DEFAULT_DISTURBANCE };
    set({ disturbance });
    get().send?.({ type: 'set_disturbance', patch: disturbance });
  },
}));

/**
 * Throttle helper for slider -> socket traffic. Dragging a slider fires far
 * faster than the engine ticks; 20 Hz is smooth to the hand and leaves the
 * socket alone. Trailing edge always fires so the final resting value is never
 * lost — the bug where you release the slider and the backend keeps the
 * second-to-last value is exactly what this guards against.
 */
export function throttle<A extends unknown[]>(
  fn: (...args: A) => void,
  ms = 50,
): (...args: A) => void {
  let last = 0;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let pending: A | null = null;

  return (...args: A) => {
    const now = Date.now();
    pending = args;
    if (now - last >= ms) {
      last = now;
      fn(...args);
      pending = null;
      return;
    }
    if (timer) return;
    timer = setTimeout(
      () => {
        timer = null;
        last = Date.now();
        if (pending) fn(...pending);
        pending = null;
      },
      ms - (now - last),
    );
  };
}
