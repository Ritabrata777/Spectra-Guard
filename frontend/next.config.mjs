/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The console is a single-page instrument; no image optimisation server needed.
  images: { unoptimized: true },
  // Standalone output keeps the PyInstaller-bundled executable self-contained:
  // the Python backend serves this build directly, so the SIH deliverable is
  // one binary with no Node runtime required on the judging machine.
  output: 'standalone',
};

export default nextConfig;
