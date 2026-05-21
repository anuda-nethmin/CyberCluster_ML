/**
 * vite.config.js — Vite Build Tool Configuration
 *
 * Vite is the development server and production bundler for this project.
 * It provides:
 *   - Instant Hot Module Replacement (HMR) during development
 *   - Optimised production builds with tree-shaking and minification
 *
 * The only plugin used is @vitejs/plugin-react, which enables:
 *   - JSX transformation (so browsers can understand .jsx files)
 *   - Fast Refresh (live component updates without losing state)
 *
 * No custom port, proxy, or alias configuration is needed because
 * the Express proxy server (server.cjs) runs separately on port 4000
 * and the React dev server defaults to port 5173.
 */

import { defineConfig } from 'vite'   // Vite helper that provides type hints
import react from '@vitejs/plugin-react' // Official React plugin for Vite

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],  // Enable React JSX + Fast Refresh support
  server: {
    port: 3000,        // Change this to whatever port you want!
    strictPort: true,  // (Optional) Forces Vite to fail if port 3000 is already in use
  }
})
