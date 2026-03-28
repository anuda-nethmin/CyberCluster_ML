/**
 * main.jsx — Application Entry Point
 *
 * This is the very first JavaScript file that runs when the app loads.
 * It mounts the root React component (<App />) into the DOM element
 * with id="root" defined in index.html.
 *
 * StrictMode is a React development helper that:
 *   - Highlights potential problems in the app
 *   - Runs certain lifecycle methods twice (dev only) to catch side-effects
 *   - Does NOT affect the production build
 */

// Import React's StrictMode wrapper for development-time checks
import { StrictMode } from 'react'

// createRoot is the React 18+ API for mounting the app (replaces ReactDOM.render)
import { createRoot } from 'react-dom/client'

// Global reset / base styles (minimal — main styles live in App.css)
import './index.css'

// The single top-level component that contains the entire application UI
import App from './App.jsx'

// Grab the <div id="root"> from index.html and render <App /> inside it
createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
