/**
 * App.jsx — Root Component of CyberCluster ML
 *
 * This single-file component contains the entire application UI.
 * It provides two workflows:
 *   1. TRAIN  — Upload a labelled CSV, configure hyper-parameters, and train
 *               a supervised regression / classification model via the C backend.
 *   2. PREDICT — Upload unlabelled CSV data and run inference using a
 *                previously trained model to predict CVSS scores or labels.
 *
 * Architecture:
 *   React (this file) → Express proxy (server.cjs:4000) → C binary (ml_engine.exe)
 *
 * Key libraries used:
 *   - React useState hook for all local state management
 *   - Recharts (LineChart) for visualising the training loss curve
 */

// ─── Imports ───────────────────────────────────────────────────────────
import { useState } from "react";   // React hook for managing component state

// Recharts — declarative charting library for React
// We only import the specific chart components we need to keep the bundle small
import {
  LineChart,           // The chart type (line graph)
  Line,                // A single data series rendered as a line
  XAxis,               // Horizontal axis configuration
  YAxis,               // Vertical axis configuration
  CartesianGrid,       // Background grid lines
  Tooltip,             // Hover-over data tooltip
  Legend,              // Chart legend (label for each line)
  ResponsiveContainer, // Wrapper that makes the chart resize with its parent
} from "recharts";

import "./App.css";  // All visual styles for this component

/* ====================================================================
 *  HELPER COMPONENT: SeverityBadge
 * ====================================================================
 *  Renders a small coloured pill/tag that indicates the severity level
 *  of a cybersecurity finding (Critical, High, Medium, Low, Info).
 *
 *  Props:
 *    severity (string) — one of "Critical", "High", "Medium", "Low", "Info"
 *
 *  The CSS class is looked up from colorMap; unknown values fall back to
 *  the "badge-info" style.
 * ================================================================== */
function SeverityBadge({ severity }) {
  // Map each severity string to its corresponding CSS class name
  const colorMap = {
    Critical: "badge-critical",   // Deep red background  (#C41E3A)
    High: "badge-high",           // Red background       (#e74c3c)
    Medium: "badge-medium",       // Orange background     (#e67e22)
    Low: "badge-low",             // Grey background       (#888)
    Info: "badge-info",           // Light-grey background (#bdc3c7)
  };

  // Render a <span> with the base "badge" class + the severity-specific class
  return <span className={`badge ${colorMap[severity] || "badge-info"}`}>{severity}</span>;
}

/* ====================================================================
 *  HELPER COMPONENT: MetricCard
 * ====================================================================
 *  A small card that displays a single highlighted metric value with a
 *  label underneath.  Used on the Training Results dashboard to show
 *  things like "Epochs Trained", "Final Training Loss", etc.
 *
 *  Props:
 *    value (string | number) — the big number / text displayed prominently
 *    label (string)          — descriptive text below the value
 * ================================================================== */
function MetricCard({ value, label }) {
  return (
    <div className="metric-card">
      {/* The large, bold number/text */}
      <div className="metric-value">{value}</div>
      {/* The small, uppercase label underneath */}
      <div className="metric-label">{label}</div>
    </div>
  );
}

/* ====================================================================
 *  MAIN COMPONENT: App
 * ====================================================================
 *  The root component rendered inside <StrictMode> by main.jsx.
 *  It manages all application state and renders three major sections:
 *    1. Sidebar  — mode switch, file upload, model config, run button
 *    2. Header   — banner with title + DMU logo
 *    3. Results  — training metrics/chart OR prediction results table
 * ================================================================== */
function App() {
  // ─── State Variables ───────────────────────────────────────────────
  // `mode` controls which workflow the user is in: "train" or "predict"
  const [mode, setMode] = useState("train");

  // The File object selected or dragged by the user, and its display name
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState("");

  // Model configuration — which algorithm and target column to use
  const [model, setModel] = useState("linear");       // "linear" or "logistic"
  const [targetCol, setTargetCol] = useState("cvss");  // "cvss" (continuous) or "label" (binary)

  // Hyper-parameters (only used during training)
  const [lr, setLr] = useState(0.01);       // Learning rate (α) — controls step size
  const [epochs, setEpochs] = useState(100); // Number of gradient-descent iterations

  // API response data
  const [trainResults, setTrainResults] = useState(null);    // JSON from /api/train
  const [predictResults, setPredictResults] = useState(null); // JSON from /api/predict

  // UI state flags
  const [loading, setLoading] = useState(false);   // true while an API call is in flight
  const [error, setError] = useState("");           // error message string (empty = no error)
  const [resetMsg, setResetMsg] = useState("");     // success message after model reset
  const [sidebarOpen, setSidebarOpen] = useState(true); // sidebar visibility (mobile toggle)

  // ─── Event Handlers ────────────────────────────────────────────────

  /**
   * handleFileChange — triggered when the user picks a file via the
   * native file-input dialog.  Stores the File object and its name in
   * state, and clears any previous error.
   */
  function handleFileChange(e) {
    const selected = e.target.files[0]; // Only take the first file
    if (selected) {
      setFile(selected);
      setFileName(selected.name);
      setError(""); // Clear old errors on new file selection
    }
  }

  /**
   * handleDrop — called when the user drags-and-drops a file onto the
   * upload zone.  Only accepts .csv files; anything else is silently
   * ignored.
   */
  function handleDrop(e) {
    e.preventDefault(); // Prevent the browser from navigating to the file
    const dropped = e.dataTransfer.files[0];
    if (dropped && dropped.name.endsWith(".csv")) {
      setFile(dropped);
      setFileName(dropped.name);
      setError("");
    }
  }

  /**
   * switchMode — safely toggles between "train" and "predict" modes.
   * Resets ALL transient state (file, results, errors) so the user
   * starts fresh in the new mode.
   */
  function switchMode(newMode) {
    setMode(newMode);
    setFile(null);
    setFileName("");
    setError("");
    setResetMsg("");
    setTrainResults(null);
    setPredictResults(null);
  }

  // ─── API Calls ─────────────────────────────────────────────────────

  /**
   * handleTrain — sends the uploaded CSV + hyper-parameter config to
   * the backend's /api/train endpoint via a multipart FormData POST.
   *
   * On success the backend returns JSON containing:
   *   { status, epochs, loss_history: [number] }
   * which is stored in `trainResults` and rendered as metric cards + chart.
   */
  async function handleTrain() {
    // Guard: ensure a file has been selected
    if (!file) { setError("Please upload a labelled CSV file for training."); return; }

    // Reset UI state before the request
    setLoading(true); setError(""); setTrainResults(null); setPredictResults(null);

    // Build a multipart/form-data body with the CSV file and all parameters
    const formData = new FormData();
    formData.append("csv", file);                       // The CSV file blob
    formData.append("model", model);                    // "linear" or "logistic"
    formData.append("target_col", targetCol);            // "cvss" or "label"
    formData.append("lr", lr.toString());                // Learning rate as string
    formData.append("epochs", epochs.toString());        // Epoch count as string

    try {
      // POST to the Express proxy server (server.cjs) on port 4000
      const response = await fetch("http://localhost:4000/api/train", {
        method: "POST",
        body: formData,  // fetch automatically sets Content-Type to multipart/form-data
      });

      const data = await response.json(); // Parse the JSON response body

      // Treat HTTP errors and explicit error statuses the same way
      if (!response.ok || data.status === "error") {
        throw new Error(data.error || data.message || "Training failed.");
      }

      setTrainResults(data); // Store the successful training results
    } catch (err) {
      setError(err.message || "Failed to train the model.");
    } finally {
      setLoading(false); // Always hide the spinner when done
    }
  }

  /**
   * handlePredict — sends the uploaded CSV to /api/predict so the
   * C backend can run inference using previously trained model weights.
   *
   * On success the backend returns JSON containing:
   *   { status, results: [ { finding_name, severity, evidence, cvss, label, predicted_cvss/label } ] }
   */
  async function handlePredict() {
    if (!file) { setError("Please upload a CSV file for inference."); return; }
    setLoading(true); setError(""); setTrainResults(null); setPredictResults(null);

    const formData = new FormData();
    formData.append("csv", file);
    formData.append("model", model);
    formData.append("target_col", targetCol);

    try {
      const response = await fetch("http://localhost:4000/api/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (!response.ok || data.status === "error") {
        throw new Error(data.error || data.message || "Prediction failed. Did you train the model first?");
      }

      setPredictResults(data); // Store the prediction results for the table
    } catch (err) {
      setError(err.message || "Failed to run prediction.");
    } finally {
      setLoading(false);
    }
  }

  /**
   * handleReset — calls /api/reset to delete the saved weights file
   * for the currently selected model.  This forces the user to retrain
   * before making new predictions.
   *
   * Sends JSON { model: "linear"|"logistic" } and shows a temporary
   * success message that auto-clears after 3 seconds.
   */
  async function handleReset() {
    setLoading(true); setResetMsg(""); setError("");
    try {
      const response = await fetch("http://localhost:4000/api/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model }), // Tell the server which model to reset
      });

      const data = await response.json();
      if (!response.ok || data.status === "error") {
        throw new Error(data.error || "Failed to reset model.");
      }

      setResetMsg(data.message); // Show success feedback
      setTimeout(() => setResetMsg(""), 3000); // Auto-dismiss after 3 s
    } catch (err) {
      setError(err.message || "Failed to reset model.");
    } finally {
      setLoading(false);
    }
  }

  // ─── Data Formatting ──────────────────────────────────────────────

  /**
   * formatLossData — transforms the raw loss_history array from the
   * training API response into the format Recharts expects:
   *   [ { epoch: 1, loss: 0.5432 }, { epoch: 2, loss: 0.4123 }, ... ]
   *
   * Each loss value is rounded to 4 decimal places for display.
   */
  const formatLossData = () => {
    if (!trainResults || !trainResults.loss_history) return [];
    return trainResults.loss_history.map((loss, i) => ({
      epoch: i + 1,                     // 1-indexed epoch number
      loss: Number(loss.toFixed(4)),     // Round to 4 d.p.
    }));
  };

  // ─── JSX Render ────────────────────────────────────────────────────
  return (
    <div className="app-container">
      {/* ==============================================================
       *  SIDEBAR — Left-hand configuration panel
       *  Contains: logo, mode tabs, file upload, model config,
       *            hyper-parameters (train only), reset button, run button
       * ============================================================== */}
      <aside className={`sidebar ${sidebarOpen ? "open" : "closed"}`}>

        {/* ── Logo & Title ── */}
        <div className="sidebar-header">
          <span className="sidebar-logo">🛡️</span>
          <span className="sidebar-title">CyberCluster ML</span>
        </div>

        <div className="sidebar-divider" />

        {/* ── Mode Tabs (Train / Predict) ──
            Clicking a tab calls switchMode() which resets all state */}
        <div className="sidebar-tabs">
          <button
            className={`tab-btn ${mode === "train" ? "active" : ""}`}
            onClick={() => switchMode("train")}
          >
            🎯 Train
          </button>
          <button
            className={`tab-btn ${mode === "predict" ? "active" : ""}`}
            onClick={() => switchMode("predict")}
          >
            🔮 Predict
          </button>
        </div>

        <div className="sidebar-divider" />

        {/* ── File Upload Section ──
            A hidden <input type="file"> is overlaid on a styled label
            so the entire drop-zone is clickable.  Drag-and-drop is
            handled by onDrop / onDragOver. */}
        <div className="sidebar-section">
          <h3 className="sidebar-section-title">📁 {mode === "train" ? "Training Data" : "Inference Data"}</h3>
          <div
            className="file-drop-zone"
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()} // Required to allow drop
          >
            {/* Invisible native file picker (positioned over the label) */}
            <input
              key={mode + (trainResults ? 't' : '') + (predictResults ? 'p' : '')}
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              id="csv-upload"
              className="file-input"
            />

            {/* Visual label — shows file name if selected, or upload prompt */}
            <label htmlFor="csv-upload" className="file-label">
              {fileName ? (
                <>
                  <span className="file-icon">📄</span>
                  <span className="file-name">{fileName}</span>
                  <span className="file-hint">Click to change</span>
                </>
              ) : (
                <>
                  <span className="file-icon">📂</span>
                  <span className="file-hint">Upload a CSV dataset</span>
                </>
              )}
            </label>
          </div>
        </div>

        <div className="sidebar-divider" />

        {/* ── Model Selection ──
            Algorithm: Linear Regression (MSE loss) or Logistic Regression (Log-Loss)
            Target Column: cvss (continuous) or label (binary) */}
        <div className="sidebar-section">
          <h3 className="sidebar-section-title">🤖 Model Setup</h3>
          <div className="param-group">
            <label className="param-label">Algorithm:</label>
            <select
              className="select-input"
              value={model}
              onChange={(e) => setModel(e.target.value)}
            >
              <option value="linear">Linear Regression (MSE)</option>
              <option value="logistic">Logistic Regression (Log-Loss)</option>
              <option value="logistic">GG (Log-Loss)</option>
            </select>
          </div>

          <div className="param-group" style={{ marginTop: "1rem" }}>
            <label className="param-label">Target Column:</label>
            <select
              className="select-input"
              value={targetCol}
              onChange={(e) => setTargetCol(e.target.value)}
            >
              <option value="cvss">CVSS Score (Continuous)</option>
              <option value="label">Label (Binary 0 / 1)</option>
            </select>
            {/* Hint reminding the user to match model type and target */}
            <span style={{ fontSize: '0.75rem', color: '#888', marginTop: '4px', display: 'block' }}>
              Ensure target matches the model type.
            </span>
          </div>
        </div>

        {/* ── Hyper-parameters (Training Only) ──
            These controls are conditionally rendered only when mode === "train".
            Learning Rate slider: 0.001 → 0.2
            Epochs slider: 10 → 2000 */}
        {mode === "train" && (
          <>
            <div className="sidebar-divider" />
            <div className="sidebar-section">
              <h3 className="sidebar-section-title">🎛️ Hyperparameters</h3>

              {/* Learning Rate (α) — step size for gradient descent */}
              <div className="param-group">
                <label className="param-label">
                  Learning Rate (α): <strong>{lr}</strong>
                </label>
                <input
                  type="range"
                  min="0.001"
                  max="0.2"
                  step="0.001"
                  value={lr}
                  onChange={(e) => setLr(Number(e.target.value))}
                  className="slider-input"
                />
              </div>

              {/* Epochs — how many times gradient descent iterates */}
              <div className="param-group">
                <label className="param-label">
                  Epochs: <strong>{epochs}</strong>
                </label>
                <input
                  type="range"
                  min="10"
                  max="2000"
                  step="10"
                  value={epochs}
                  onChange={(e) => setEpochs(Number(e.target.value))}
                  className="slider-input"
                />
              </div>
            </div>
          </>
        )}

        <div className="sidebar-divider" />

        {/* ── Reset Model Button ──
            Calls handleReset() → POST /api/reset to delete saved weights */}
        <div className="sidebar-section">
          <button
            className="reset-button"
            onClick={handleReset}
            disabled={loading}
            style={{ width: '100%', padding: '0.75rem', background: 'transparent', border: '1px solid #C41E3A', color: '#C41E3A', borderRadius: '4px', cursor: 'pointer', marginBottom: '1rem', fontWeight: 'bold' }}
          >
            🗑️ Clear {model === "linear" ? "Linear" : "Logistic"} Model Memory
          </button>
          {/* Temporary success message — auto-clears after 3 seconds */}
          {resetMsg && <div style={{ color: '#27ae60', fontSize: '0.8rem', textAlign: 'center', marginBottom: '1rem' }}>{resetMsg}</div>}
        </div>

        {/* ── Primary Action Button ──
            Dynamically calls handleTrain or handlePredict depending on mode.
            Shows a spinner icon when loading. */}
        <button
          className="run-button"
          onClick={mode === "train" ? handleTrain : handlePredict}
          disabled={loading}
        >
          {loading ? "⏳ Processing..." : mode === "train" ? "🚀 Start Training" : "🔮 Run Prediction"}
        </button>
      </aside>

      {/* ── Mobile Sidebar Toggle ──
          A small floating button visible only on screens ≤ 768 px.
          Toggles the sidebar open/closed. */}
      <button className="sidebar-toggle" onClick={() => setSidebarOpen(!sidebarOpen)}>
        {sidebarOpen ? "✕" : "☰"}
      </button>

      {/* ==============================================================
       *  MAIN CONTENT AREA — Right of sidebar
       * ============================================================== */}
      <main className="main-content">

        {/* ── Header Banner ──
            Red banner with app title, subtitle, and DMU university logo.
            The .header-glow div and the ::after pseudo-element create
            subtle radial-gradient light effects for visual depth. */}
        <div className="header-banner">
          <div className="header-glow" />
          <div className="header-text">
            <h1>CyberCluster ML</h1>
            <p>Supervised Regression Learning &amp; Threat Intelligence Inference</p>
          </div>
          <img
            src="https://www.dmu.ac.uk/siteelements/2018/images/logo.svg"
            alt="De Montfort University Logo"
            className="dmu-logo"
          />
        </div>

        {/* ── Error Message ──
            Conditionally rendered red banner when `error` is non-empty */}
        {error && <div className="error-message">{error}</div>}

        {/* ── Loading Spinner ──
            Shown while an API request is in flight.  The CSS @keyframes
            "spin" animation rotates the .spinner div continuously. */}
        {loading && (
          <div className="loading-container">
            <div className="spinner" />
            <p>{mode === "train" ? "Training model weights via gradient descent..." : "Executing inference..."}</p>
          </div>
        )}

        {/* ─────────────────────────────────────────────────────────────
         *  TRAINING RESULTS SECTION
         *  Rendered only when: mode === "train" AND trainResults exists
         *  AND we're not currently loading.
         *
         *  Contains:
         *    1. Four MetricCards showing key stats
         *    2. A Recharts LineChart plotting the loss curve
         * ───────────────────────────────────────────────────────────── */}
        {mode === "train" && trainResults && !loading && (
          <>
            <div className="section-divider" />
            <div className="section-header">
              <h2 className="section-title">📈 Training Results</h2>
              <p className="section-subtitle">Model weights saved successfully</p>
            </div>

            {/* Metric cards — 4-column grid on desktop, 2-col on tablet, 1-col on mobile */}
            <div className="metrics-grid">
              <MetricCard value={trainResults.epochs} label="Epochs Trained" />
              <MetricCard
                value={trainResults.loss_history[trainResults.loss_history.length - 1].toFixed(4)}
                label="Final Training Loss"
              />
              <MetricCard value={model.toUpperCase()} label="Model Architecture" />
              <MetricCard value={targetCol} label="Optimized Target" />
            </div>

            {/* Loss Curve Chart — shows how the loss decreased over epochs */}
            <div className="section-header">
              <h2 className="section-title">📉 Loss Curve (Gradient Descent)</h2>
            </div>
            <div className="chart-card" style={{ height: "400px" }}>
              {/* ResponsiveContainer makes the chart fill its parent's width */}
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={formatLossData()} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                  {/* Dashed background grid */}
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  {/* X-axis: epoch number */}
                  <XAxis dataKey="epoch" name="Epoch" stroke="#888" label={{ value: 'Epochs', position: 'insideBottom', offset: -10 }} />
                  {/* Y-axis: loss value */}
                  <YAxis stroke="#888" label={{ value: 'Loss (MSE / Log-Loss)', angle: -90, position: 'insideLeft', offset: -10 }} />
                  {/* Tooltip appears on hover with styled card */}
                  <Tooltip
                    contentStyle={{
                      background: "#fff",
                      border: "1px solid #eee",
                      borderRadius: "8px",
                      boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
                    }}
                  />
                  {/* Legend label for the line */}
                  <Legend verticalAlign="top" />
                  {/* The actual loss line — maroon colour, no individual dots */}
                  <Line type="monotone" dataKey="loss" stroke="#C41E3A" strokeWidth={3} dot={false} activeDot={{ r: 8 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </>
        )}

        {/* ─────────────────────────────────────────────────────────────
         *  PREDICTION RESULTS SECTION
         *  Rendered only when: mode === "predict" AND predictResults exists
         *
         *  Contains a table of all findings with:
         *    - Finding name, severity badge, evidence text
         *    - Actual value and the model's predicted value
         * ───────────────────────────────────────────────────────────── */}
        {mode === "predict" && predictResults && !loading && (
          <>
            <div className="section-divider" />
            <div className="section-header">
              <h2 className="section-title">📊 Inference Results</h2>
              <p className="section-subtitle">Predictions generated using pre-trained weights</p>
            </div>

            <div className="cluster-card" style={{ marginTop: "2rem" }}>
              <div className="cluster-header" style={{ cursor: "default" }}>
                <div className="cluster-title-row">
                  <h3>Prediction output mapping for {predictResults.results.length} findings</h3>
                </div>
              </div>

              {/* Scrollable table container for horizontal overflow on small screens */}
              <div className="findings-table-container">
                <table className="findings-table">
                  <thead>
                    <tr>
                      <th>Finding</th>
                      <th>Severity</th>
                      <th>Evidence</th>
                      {/* Column header changes based on the target column */}
                      <th>{targetCol === "cvss" ? "Actual CVSS" : "Actual Label"}</th>
                      <th style={{ color: "#C41E3A" }}>Predicted {targetCol}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {/* Iterate over each prediction result and render a table row */}
                    {predictResults.results.map((f, idx) => (
                      <tr key={idx}>
                        <td className="finding-name">{f.finding_name}</td>
                        <td><SeverityBadge severity={f.severity} /></td>
                        <td className="evidence-cell">{f.evidence}</td>
                        {/* Show actual value for comparison */}
                        <td>{targetCol === "cvss" ? f.cvss : f.label}</td>
                        {/* Show the model's predicted value, rounded to 3 d.p. */}
                        <td style={{ fontWeight: "bold" }}>
                          {Number(f[`predicted_${targetCol}`]).toFixed(3)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}

        {/* ─────────────────────────────────────────────────────────────
         *  EMPTY STATE
         *  Shown when there are no results and nothing is loading.
         *  Acts as a welcome screen with instructions for the user.
         * ───────────────────────────────────────────────────────────── */}
        {!trainResults && !predictResults && !loading && (
          <div className="empty-state">
            <div className="icon-circle">🛡️</div>
            <h2>
              Welcome to <span className="accent">Supervised</span> CyberCluster
            </h2>
            <p>
              {mode === "train"
                ? "Upload historical labelled data and configure your regression model to learn attack patterns."
                : "Upload new data and apply your pre-trained model engine to infer threats automatically."}
            </p>
          </div>
        )}
      </main>
    </div>
  );
}

// Export App as the default export so main.jsx can import it
export default App;
