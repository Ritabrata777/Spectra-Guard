'use client';

/**
 * SPECTRA GUARD — Terminal & event log.
 *
 * Monospaced system output read through deeply tinted glass. Two decisions here
 * are worth stating:
 *
 * RENDERING. Log lines arrive in bursts — an acquisition sequence can emit a
 * dozen events inside a few frames. The bus coalesces its notifications to 10 Hz
 * and this component only ever renders the tail window, so a burst costs one
 * render of ~120 rows rather than twelve renders of the whole buffer.
 *
 * AUTOSCROLL. It follows the tail, but stops the moment the operator scrolls up,
 * and says so. Auto-scrolling a log out from under someone who is reading it is
 * one of the most reliably infuriating things software does; a small "following /
 * paused" annunciator costs nothing and removes the problem.
 */

import { motion } from 'framer-motion';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { GlassPanel } from '@/components/glass/GlassPanel';
import { bus } from '@/lib/telemetry-bus';
import { useCommandStore } from '@/lib/store';
import { useSlowTelemetry, type PatLink } from '@/lib/use-pat-link';
import type { LogEvent, LogLevel } from '@/lib/types';

/** Tail window. Deep enough to scroll back through an acquisition, bounded. */
const TAIL = 160;

const LEVEL_STYLE: Record<LogLevel, { text: string; tag: string }> = {
  INFO: { text: 'text-readout-secondary', tag: 'text-readout-tertiary' },
  OK: { text: 'text-photon-200', tag: 'text-photon-600' },
  WARN: { text: 'text-ember-400', tag: 'text-ember-600' },
  ERROR: { text: 'text-plasma-400', tag: 'text-plasma-600' },
};

function LogLine({ e }: { e: LogEvent }) {
  const s = LEVEL_STYLE[e.level];
  return (
    <div className="flex gap-3 whitespace-pre-wrap px-1 leading-[1.55]">
      <span className="w-[3.75rem] shrink-0 text-right text-readout-dim">{e.t.toFixed(3)}</span>
      <span className={`w-[2.75rem] shrink-0 uppercase ${s.tag}`}>{e.level}</span>
      <span className="w-[3.25rem] shrink-0 uppercase text-readout-tertiary">{e.channel}</span>
      <span className={s.text}>{e.message}</span>
    </div>
  );
}

export function TerminalLog({ link }: { link: PatLink }) {
  const { logCount } = useSlowTelemetry();
  const expanded = useCommandStore((s) => s.terminalExpanded);
  const setExpanded = useCommandStore((s) => s.setTerminalExpanded);

  const scrollRef = useRef<HTMLDivElement>(null);
  const [following, setFollowing] = useState(true);
  const [levelFilter, setLevelFilter] = useState<LogLevel | 'ALL'>('ALL');

  // `logCount` in the dependency list is what makes this recompute — the array
  // itself is mutated in place by the bus, so identity is not a useful signal.
  const lines = useMemo(() => {
    const all = levelFilter === 'ALL' ? bus.logs : bus.logs.filter((l) => l.level === levelFilter);
    return all.slice(-TAIL);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [logCount, levelFilter]);

  const onScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    // 24 px of slack: "close enough to the bottom" should not require pixel
    // precision, and momentum scrolling rarely lands exactly at zero.
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
    setFollowing(atBottom);
  }, []);

  useEffect(() => {
    if (!following) return;
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [lines, following]);

  const counts = useMemo(() => {
    let warn = 0;
    let err = 0;
    for (const l of bus.logs) {
      if (l.level === 'WARN') warn++;
      else if (l.level === 'ERROR') err++;
    }
    return { warn, err };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [logCount]);

  return (
    <GlassPanel
      flush
      blur="deep"
      className="z-panel shrink-0"
      label="Terminal · system events"
      headerSlot={
        <div className="flex items-center gap-2">
          {counts.warn > 0 && (
            <span className="rounded-pill border border-ember/40 bg-ember/[0.10] px-2 py-0.5 font-mono text-[0.5625rem] uppercase tracking-[0.12em] text-ember-400">
              {counts.warn} warn
            </span>
          )}
          {counts.err > 0 && (
            <span className="rounded-pill border border-plasma/45 bg-plasma/[0.12] px-2 py-0.5 font-mono text-[0.5625rem] uppercase tracking-[0.12em] text-plasma-400">
              {counts.err} err
            </span>
          )}

          <div className="tactile flex items-center gap-0.5 rounded-pill p-0.5">
            {(['ALL', 'OK', 'WARN', 'ERROR'] as const).map((lv) => (
              <button
                key={lv}
                type="button"
                onClick={() => setLevelFilter(lv)}
                aria-pressed={levelFilter === lv}
                className={[
                  'rounded-pill px-2 py-0.5 font-mono text-[0.5625rem] uppercase tracking-[0.12em] transition-colors duration-200',
                  levelFilter === lv
                    ? 'bg-white/[0.12] text-readout-primary'
                    : 'text-readout-tertiary hover:text-readout-secondary',
                ].join(' ')}
              >
                {lv}
              </button>
            ))}
          </div>

          <span
            className={[
              'font-mono text-[0.5625rem] uppercase tracking-[0.14em]',
              following ? 'text-photon-600' : 'text-ember-600',
            ].join(' ')}
          >
            {following ? 'following' : 'paused'}
          </span>

          <button
            type="button"
            onClick={link.exportCsv}
            className="rounded-pill border border-white/12 bg-white/[0.05] px-2.5 py-0.5 font-mono text-[0.5625rem] uppercase tracking-[0.12em] text-readout-secondary transition-colors duration-200 hover:border-photon/40 hover:text-photon-200"
          >
            export csv
          </button>

          <button
            type="button"
            onClick={() => setExpanded(!expanded)}
            aria-expanded={expanded}
            className="rounded-pill border border-white/12 bg-white/[0.05] px-2.5 py-0.5 font-mono text-[0.5625rem] uppercase tracking-[0.12em] text-readout-secondary transition-colors duration-200 hover:border-photon/40 hover:text-photon-200"
          >
            {expanded ? 'collapse' : 'expand'}
          </button>
        </div>
      }
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1], delay: 0.12 }}
    >
      <motion.div
        className="glass-terminal relative overflow-hidden"
        animate={{ height: expanded ? 288 : 132 }}
        transition={{ duration: 0.45, ease: [0.32, 0.72, 0, 1] }}
      >
        <div
          ref={scrollRef}
          onScroll={onScroll}
          className="scroll-frost h-full overflow-y-auto px-3 py-2 font-mono text-[0.6875rem] tabular-nums"
        >
          {lines.length === 0 ? (
            <div className="px-1 py-1 text-readout-dim">
              no events{levelFilter !== 'ALL' ? ` at level ${levelFilter}` : ''} — arm a run to begin
            </div>
          ) : (
            lines.map((e, i) => <LogLine key={`${e.seq}-${e.t}-${i}`} e={e} />)
          )}
        </div>

        {/* Top fade so scrolled-out lines dissolve into the glass rather than
            being guillotined by the panel edge. */}
        <div
          aria-hidden
          className="pointer-events-none absolute inset-x-0 top-0 h-6"
          style={{ background: 'linear-gradient(to bottom, rgba(5,7,11,0.85), rgba(5,7,11,0))' }}
        />

        {!following && (
          <button
            type="button"
            onClick={() => {
              setFollowing(true);
              const el = scrollRef.current;
              if (el) el.scrollTop = el.scrollHeight;
            }}
            className="absolute bottom-2 right-3 rounded-pill border border-photon/40 bg-photon/[0.14] px-2.5 py-0.5 font-mono text-[0.5625rem] uppercase tracking-[0.12em] text-photon-50 shadow-glow-photon"
          >
            jump to latest
          </button>
        )}
      </motion.div>
    </GlassPanel>
  );
}
