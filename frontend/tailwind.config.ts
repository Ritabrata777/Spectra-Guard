import type { Config } from 'tailwindcss';

/**
 * SPECTRA GUARD liquid-glass theme.
 *
 * Palette discipline: three accents only — photon (lock), ember (degraded),
 * plasma (fault) — over a two-step obsidian/abyss void. Every glass surface is
 * a *white* film at low alpha, never a grey fill: real frosted glass tints
 * toward the light behind it, and greys read as plastic.
 */
const config: Config = {
  content: ['./app/**/*.{ts,tsx}', './components/**/*.{ts,tsx}', './lib/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        /* Void — what the glass floats above. */
        obsidian: '#05070B',
        abyss: '#0A0F17',
        hull: '#111826',

        /* Target lock / nominal. */
        photon: {
          DEFAULT: '#22E0FF',
          50: '#EAFBFF',
          200: '#9FF0FF',
          400: '#4CE6FF',
          500: '#22E0FF',
          600: '#0FB4D4',
          900: '#083D4A',
        },

        /* Degraded / coasting / caution. */
        ember: {
          DEFAULT: '#FFA92B',
          400: '#FFC062',
          500: '#FFA92B',
          600: '#D9821A',
        },

        /* Loss of lock / fault. */
        plasma: {
          DEFAULT: '#FF3B5C',
          400: '#FF7089',
          500: '#FF3B5C',
          600: '#D91F40',
        },

        /* Instrument typography. */
        readout: {
          primary: '#E8EEF7',
          secondary: '#94A6BC',
          tertiary: '#5C6E85',
          dim: '#3A485A',
        },
      },

      fontFamily: {
        /* Display: technical grotesque with real character in the g/a/y. */
        display: ['"Space Grotesk"', 'system-ui', 'sans-serif'],
        /* Body/UI. */
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        /* All numerics. Plex Mono is the authentic instrumentation face and
           ships true tabular figures, so digits never dance as values change. */
        mono: ['"IBM Plex Mono"', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },

      fontSize: {
        /* Eyebrow labels — small, wide-tracked, uppercase. */
        eyebrow: ['0.625rem', { lineHeight: '0.75rem', letterSpacing: '0.18em' }],
        micro: ['0.6875rem', { lineHeight: '0.875rem', letterSpacing: '0.06em' }],
        readout: ['0.8125rem', { lineHeight: '1.125rem', letterSpacing: '0.02em' }],
        /* Big telemetry numerals. */
        numeral: ['1.75rem', { lineHeight: '1.875rem', letterSpacing: '-0.02em' }],
        numeralLg: ['2.5rem', { lineHeight: '2.5rem', letterSpacing: '-0.03em' }],
      },

      borderRadius: {
        glass: '1.25rem',
        pill: '999px',
      },

      backdropBlur: {
        frost: '24px',
        deep: '40px',
        abyssal: '64px',
      },

      boxShadow: {
        /* The three-part glass recipe: cast shadow for lift, inset top
           highlight for the specular edge, inset bottom for thickness. */
        glass:
          '0 24px 64px -16px rgba(0,0,0,0.75), inset 0 1px 0 0 rgba(255,255,255,0.14), inset 0 -1px 0 0 rgba(255,255,255,0.04)',
        'glass-sm':
          '0 8px 24px -8px rgba(0,0,0,0.6), inset 0 1px 0 0 rgba(255,255,255,0.12)',
        /* Floating pill sits closer to the viewer, so a longer, softer cast. */
        pill:
          '0 32px 80px -20px rgba(0,0,0,0.85), inset 0 1px 0 0 rgba(255,255,255,0.18), inset 0 -1px 0 0 rgba(255,255,255,0.05)',
        'glow-photon': '0 0 24px -2px rgba(34,224,255,0.55), 0 0 64px -12px rgba(34,224,255,0.35)',
        'glow-ember': '0 0 24px -2px rgba(255,169,43,0.55), 0 0 64px -12px rgba(255,169,43,0.32)',
        'glow-plasma': '0 0 24px -2px rgba(255,59,92,0.6), 0 0 64px -12px rgba(255,59,92,0.35)',
        /* Pressed state for tactile controls. */
        'inset-tactile':
          'inset 0 2px 6px 0 rgba(0,0,0,0.6), inset 0 -1px 0 0 rgba(255,255,255,0.06)',
      },

      /* z-index is a named scale, not ad-hoc integers. Glass layering breaks
         the moment two surfaces disagree about who is in front. */
      zIndex: {
        scene: '0',
        plot: '10',
        panel: '20',
        bezel: '30',
        float: '40',
        overlay: '50',
        modal: '60',
      },

      transitionTimingFunction: {
        /* Apple's standard ease — the reason their panels feel like matter. */
        liquid: 'cubic-bezier(0.32, 0.72, 0, 1)',
        settle: 'cubic-bezier(0.16, 1, 0.3, 1)',
      },

      keyframes: {
        'scan-sweep': {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(200%)' },
        },
        'breathe': {
          '0%, 100%': { opacity: '0.55' },
          '50%': { opacity: '1' },
        },
        'reticle-lock': {
          '0%': { transform: 'scale(1.4)', opacity: '0' },
          '100%': { transform: 'scale(1)', opacity: '1' },
        },
      },
      animation: {
        'scan-sweep': 'scan-sweep 2.4s cubic-bezier(0.32,0.72,0,1) infinite',
        breathe: 'breathe 2s ease-in-out infinite',
        'reticle-lock': 'reticle-lock 0.35s cubic-bezier(0.16,1,0.3,1) both',
      },
    },
  },
  plugins: [],
};

export default config;
