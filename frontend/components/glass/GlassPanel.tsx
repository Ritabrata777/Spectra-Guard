'use client';

import { motion, type HTMLMotionProps } from 'framer-motion';
import { forwardRef, type ReactNode } from 'react';

export type GlassBlur = 'frost' | 'deep' | 'abyssal';
export type GlassAccent = 'none' | 'photon' | 'ember' | 'plasma';

const BLUR_CLASS: Record<GlassBlur, string> = {
  frost: '',
  deep: 'glass-deep',
  abyssal: 'glass-abyssal',
};

/**
 * Accent rings are drawn as a *border tint plus outer glow* rather than a solid
 * stroke, so the edge still reads as glass catching coloured light instead of a
 * painted outline.
 */
const ACCENT_CLASS: Record<GlassAccent, string> = {
  none: '',
  photon: 'border-photon/35 shadow-glow-photon',
  ember: 'border-ember/40 shadow-glow-ember',
  plasma: 'border-plasma/45 shadow-glow-plasma',
};

export interface GlassPanelProps extends Omit<HTMLMotionProps<'section'>, 'children'> {
  children?: ReactNode;
  /** Eyebrow label rendered in the panel's header rail. */
  label?: string;
  /** Right-aligned slot in the header rail — status chips, counters, actions. */
  headerSlot?: ReactNode;
  blur?: GlassBlur;
  accent?: GlassAccent;
  /** Travelling highlight along the top bevel. On by default; it is the tell. */
  specular?: boolean;
  /** Subsurface colour bleeding up through the glass from behind. */
  caustic?: boolean;
  /** Removes internal padding for panels whose child owns the full bleed
      (canvas viewports, plots). */
  flush?: boolean;
  className?: string;
  bodyClassName?: string;
}

/**
 * The single frosted surface every panel in the console is built from.
 *
 * Centralising it matters for more than tidiness: glass only looks like one
 * material if every sheet shares the same blur radius, film gradient, edge
 * alpha, and shadow falloff. Ten components each hand-rolling `backdrop-blur`
 * is how a "glassmorphic" UI ends up looking like ten different plastics.
 */
export const GlassPanel = forwardRef<HTMLElement, GlassPanelProps>(function GlassPanel(
  {
    children,
    label,
    headerSlot,
    blur = 'frost',
    accent = 'none',
    specular = true,
    caustic = false,
    flush = false,
    className = '',
    bodyClassName = '',
    ...motionProps
  },
  ref,
) {
  return (
    <motion.section
      ref={ref}
      className={[
        'glass relative isolate overflow-hidden rounded-glass',
        BLUR_CLASS[blur],
        ACCENT_CLASS[accent],
        specular ? 'specular' : '',
        caustic ? 'caustic' : '',
        // Accent transitions are slow and eased so a stage change feels like
        // light moving through the panel, not a class swap.
        'transition-[border-color,box-shadow] duration-500 ease-liquid',
        className,
      ]
        .filter(Boolean)
        .join(' ')}
      {...motionProps}
    >
      {(label || headerSlot) && (
        <header className="flex items-center justify-between gap-3 border-b border-white/[0.07] px-4 py-2.5">
          {label ? <span className="eyebrow text-crisp">{label}</span> : <span />}
          {headerSlot}
        </header>
      )}
      <div className={[flush ? '' : 'p-4', 'relative', bodyClassName].filter(Boolean).join(' ')}>
        {children}
      </div>
    </motion.section>
  );
});

/**
 * Small status chip for panel header rails. Kept here beside GlassPanel because
 * its alpha values are tuned against the same film.
 */
export function StatusChip({
  tone = 'neutral',
  children,
  pulse = false,
}: {
  tone?: 'neutral' | 'photon' | 'ember' | 'plasma';
  children: ReactNode;
  pulse?: boolean;
}) {
  const tones = {
    neutral: 'border-white/12 bg-white/[0.06] text-readout-secondary',
    photon: 'border-photon/40 bg-photon/[0.12] text-photon-200',
    ember: 'border-ember/45 bg-ember/[0.12] text-ember-400',
    plasma: 'border-plasma/50 bg-plasma/[0.14] text-plasma-400',
  } as const;

  const dot = {
    neutral: 'bg-readout-tertiary',
    photon: 'bg-photon shadow-[0_0_8px_rgba(34,224,255,0.9)]',
    ember: 'bg-ember shadow-[0_0_8px_rgba(255,169,43,0.9)]',
    plasma: 'bg-plasma shadow-[0_0_8px_rgba(255,59,92,0.9)]',
  } as const;

  return (
    <span
      className={[
        'inline-flex items-center gap-1.5 rounded-pill border px-2 py-0.5',
        'font-mono text-eyebrow uppercase',
        tones[tone],
      ].join(' ')}
    >
      <span
        className={[
          'h-1.5 w-1.5 rounded-full',
          dot[tone],
          pulse ? 'animate-breathe' : '',
        ].join(' ')}
      />
      {children}
    </span>
  );
}
