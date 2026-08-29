'use client';

/**
 * SPECTRA GUARD — console shell.
 *
 * Z-LAYERING, which is the thing that most often ruins a glass UI:
 *
 *   scene (0)   the live viewport canvas
 *   plot (10)   the trajectory panel
 *   panel (20)  command bar, sidebar, terminal — the primary glass sheets
 *   bezel (30)  the inset ring drawn over the viewport
 *   float (40)  the disturbance injector pill, which hovers above everything
 *   overlay(50) in-viewport annunciators (the coast banner)
 *   modal (60)  reserved
 *
 * Named steps rather than ad-hoc integers, because a backdrop-filter only blurs
 * what is painted *behind* it — so the moment two sheets disagree about their
 * order, one of them stops refracting and instantly reads as flat plastic. The
 * scale lives in tailwind.config.ts so there is exactly one place to reason about.
 *
 * The link hook is called HERE and once. It owns the socket, the fallback engine,
 * and the export path; passing the returned handle down is deliberate, so there
 * is no possibility of two components each opening a WebSocket.
 */

import { useEffect } from 'react';
import { AcquisitionCommandBar } from '@/components/command/AcquisitionCommandBar';
import { TerminalLog } from '@/components/command/TerminalLog';
import { DisturbanceInjector } from '@/components/controls/DisturbanceInjector';
import { SimpleReadout } from '@/components/readout/SimpleReadout';
import { OpticalTrackingArray } from '@/components/tracking/OpticalTrackingArray';
import { useCommandStore } from '@/lib/store';
import { usePatLink } from '@/lib/use-pat-link';

export default function ConsolePage() {
  const link = usePatLink();
  const setInjectorOpen = useCommandStore((s) => s.setInjectorOpen);
  const injectorOpen = useCommandStore((s) => s.injectorOpen);

  /**
   * Operator shortcuts.
   *
   * Not a gimmick: during a five-minute demo slot, reaching for a pill in the
   * corner of a projected screen with a trackpad costs real seconds. Guarded
   * against firing while a control has focus so Space still activates buttons.
   */
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
      if (t && (t.tagName === 'BUTTON' || t.getAttribute('role') === 'slider')) {
        if (e.code === 'Space') return;
      }
      if (e.code === 'Space') {
        e.preventDefault();
        if (link.running) link.stop();
        else link.start();
      } else if (e.key === 'd' || e.key === 'D') {
        setInjectorOpen(!injectorOpen);
      } else if (e.key === 'e' || e.key === 'E') {
        link.exportCsv();
      } else if (e.key === 'r' || e.key === 'R') {
        link.reset();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [link, injectorOpen, setInjectorOpen]);

  return (
    <main className="flex h-screen flex-col gap-3 overflow-hidden p-4">
      <AcquisitionCommandBar link={link} />

      {/* The console proper. `min-h-0` on both the row and its children is what
          lets the viewport actually shrink inside a flex column instead of
          overflowing the screen — the single most common flexbox mistake in
          full-height dashboards. */}
      <div className="grid min-h-0 flex-1 grid-cols-1 gap-3 xl:grid-cols-[minmax(0,1fr)_23.5rem]">
        <OpticalTrackingArray className="min-h-0" />
        <SimpleReadout className="z-panel min-h-0" />
      </div>

      <TerminalLog link={link} />

      <DisturbanceInjector />

      {/* Shortcut legend, low-contrast: discoverable without competing with
          telemetry for attention. */}
      <div className="pointer-events-none fixed bottom-5 left-5 z-float hidden font-mono text-[0.5625rem] uppercase tracking-[0.14em] text-readout-dim lg:block">
        space arm/hold · d disturbance · e export · r reset
      </div>
    </main>
  );
}
