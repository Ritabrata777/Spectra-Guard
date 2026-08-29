import type { Metadata, Viewport } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Spectra Guard · Coarse PAT Console',
  description:
    'AI-based virtual camera tracking system for coarse alignment of mobile free-space optical communication terminals. YOLOv11 + EKF fusion with physics-informed atmospheric channel simulation.',
  applicationName: 'Spectra Guard',
  authors: [{ name: 'Team Incu3bit' }],
};

export const viewport: Viewport = {
  themeColor: '#05070B',
  // The console is a fixed-layout instrument panel; letting the browser zoom the
  // viewport on a phone would break the glass layering for no benefit.
  width: 'device-width',
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <head>
        {/*
          Fonts are loaded via <link> rather than `next/font/google`.
          next/font fetches the font files at BUILD time, which means an offline
          or air-gapped build — exactly the situation when packaging the SIH
          standalone executable on a machine with no internet — fails outright.
          A stylesheet link degrades instead: the Tailwind font stacks fall back
          to system-ui / ui-monospace and the console still renders correctly.
        */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="min-h-screen bg-obsidian font-sans text-readout-primary antialiased">
        {children}
      </body>
    </html>
  );
}
