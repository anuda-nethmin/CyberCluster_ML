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
  const [model, setModel] = useState("linear");       // "linear", "logistic", "knn", "kmeans", or "dtree"
  const [targetCol, setTargetCol] = useState("cvss");  // "cvss" (continuous) or "label" (binary)

  // Hyper-parameters — Linear / Logistic (gradient descent models)
  const [lr, setLr] = useState(0.01);       // Learning rate (α) — controls step size
  const [epochs, setEpochs] = useState(100); // Number of gradient-descent iterations

  // Hyper-parameters — KNN
  const [k, setK] = useState(5);             // Number of nearest neighbours to consider

  // Hyper-parameters — K-Means
  const [kClusters, setKClusters] = useState(3);  // Number of clusters to partition data into
  const [maxIter, setMaxIter] = useState(100);      // Maximum iterations for convergence

  // Hyper-parameters — Decision Tree
  const [maxDepth, setMaxDepth] = useState(5);  // Maximum depth of the tree (controls overfitting)

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
    // Clear the input value so the same file can be selected again without needing a refresh
    e.target.value = null;
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
    formData.append("model", model);                    // "linear", "logistic", "knn", "kmeans", or "dtree"
    formData.append("target_col", targetCol);            // "cvss" or "label"

    // Model-specific hyperparameters appended to the form data
    if (model === "knn") {
      formData.append("k", k.toString());
      formData.append("lr", "0");       // Dummy value — ignored by backend for KNN
      formData.append("epochs", "1");   // Dummy value — ignored by backend for KNN
    } else if (model === "kmeans") {
      formData.append("lr", kClusters.toString());   // lr position carries k_clusters
      formData.append("epochs", maxIter.toString()); // epochs position carries max_iter
    } else if (model === "dtree") {
      formData.append("lr", maxDepth.toString());    // lr position carries max_depth
      formData.append("epochs", "1");                // epochs ignored for dtree
    } else {
      formData.append("lr", lr.toString());                // Learning rate as string
      formData.append("epochs", epochs.toString());        // Epoch count as string
    }

    try {
      // POST to the Express proxy server (server.cjs) on port 4000
      const apiUrl = import.meta.env.DEV ? "http://localhost:4000" : "";
      const response = await fetch(`${apiUrl}/api/train`, {
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

    // KNN and K-Means need k at prediction time too
    if (model === "knn") {
      formData.append("k", k.toString());
    } else if (model === "kmeans") {
      formData.append("k", kClusters.toString());
    }

    try {
      const apiUrl = import.meta.env.DEV ? "http://localhost:4000" : "";
      const response = await fetch(`${apiUrl}/api/predict`, {
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
   * Sends JSON { model: "linear"|"logistic"|"knn"|"kmeans"|"dtree" }
   * and shows a temporary success message that auto-clears after 3 seconds.
   */
  async function handleReset() {
    setLoading(true); setResetMsg(""); setError("");
    try {
      const apiUrl = import.meta.env.DEV ? "http://localhost:4000" : "";
      const response = await fetch(`${apiUrl}/api/reset`, {
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
              key={mode}
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
            Algorithm: Linear, Logistic, KNN, K-Means, or Decision Tree
            Target Column: cvss (continuous) or label (binary) */}
        <div className="sidebar-section">
          <h3 className="sidebar-section-title">🤖 Model Setup</h3>
          <div className="param-group">
            <label className="param-label">Algorithm:</label>
            <select
              className="select-input"
              value={model}
              onChange={(e) => {
                const newModel = e.target.value;
                setModel(newModel);
                // KNN, K-Means, and Decision Tree only support label target
                if (newModel === "knn" || newModel === "kmeans" || newModel === "dtree") setTargetCol("label");
              }}
            >
              <option value="linear">Linear Regression (MSE)</option>
              <option value="logistic">Logistic Regression (Log-Loss)</option>
              <option value="knn">K-Nearest Neighbors (KNN)</option>
              <option value="kmeans">K-Means Clustering</option>
              <option value="dtree">Decision Tree</option>
            </select>
          </div>

          <div className="param-group" style={{ marginTop: "1rem" }}>
            <label className="param-label">Target Column:</label>
            <select
              className="select-input"
              value={targetCol}
              onChange={(e) => setTargetCol(e.target.value)}
              disabled={model === "knn" || model === "kmeans" || model === "dtree"}
            >
              <option value="cvss">CVSS Score (Continuous)</option>
              <option value="label">Label (Binary 0 / 1)</option>
            </select>
            {/* Hint reminding the user to match model type and target */}
            <span style={{ fontSize: '0.75rem', color: '#888', marginTop: '4px', display: 'block' }}>
              {model === "knn"
                ? "KNN uses binary classification — target locked to 'label'."
                : model === "kmeans"
                  ? "K-Means is unsupervised — groups findings into clusters."
                  : model === "dtree"
                    ? "Decision Tree uses binary classification — target locked to 'label'."
                    : "Ensure target matches the model type."}
            </span>
          </div>
        </div>

        {/* ── Hyper-parameters (Training Only) ──
            These controls are conditionally rendered only when mode === "train".
            Each model type shows its own relevant hyper-parameter controls. */}
        {mode === "train" && (
          <>
            <div className="sidebar-divider" />
            <div className="sidebar-section">
              <h3 className="sidebar-section-title">🎛️ Hyperparameters</h3>

              {model === "knn" ? (
                /* KNN-specific: only K (number of neighbors) */
                <div className="param-group">
                  <label className="param-label">
                    K (Neighbors): <strong>{k}</strong>
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="25"
                    step="2"
                    value={k}
                    onChange={(e) => setK(Number(e.target.value))}
                    className="slider-input"
                  />
                  <span style={{ fontSize: '0.75rem', color: '#888', marginTop: '4px', display: 'block' }}>
                    Use odd numbers to avoid ties in voting.
                  </span>
                </div>
              ) : model === "kmeans" ? (
                /* K-Means: K clusters + max iterations */
                <>
                  <div className="param-group">
                    <label className="param-label">
                      K (Clusters): <strong>{kClusters}</strong>
                    </label>
                    <input
                      type="range"
                      min="2"
                      max="10"
                      step="1"
                      value={kClusters}
                      onChange={(e) => setKClusters(Number(e.target.value))}
                      className="slider-input"
                    />
                    <span style={{ fontSize: '0.75rem', color: '#888', marginTop: '4px', display: 'block' }}>
                      Number of groups to partition data into.
                    </span>
                  </div>
                  <div className="param-group">
                    <label className="param-label">
                      Max Iterations: <strong>{maxIter}</strong>
                    </label>
                    <input
                      type="range"
                      min="10"
                      max="200"
                      step="10"
                      value={maxIter}
                      onChange={(e) => setMaxIter(Number(e.target.value))}
                      className="slider-input"
                    />
                  </div>
                </>
              ) : model === "dtree" ? (
                /* Decision Tree: max depth */
                <div className="param-group">
                  <label className="param-label">
                    Max Depth: <strong>{maxDepth}</strong>
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="20"
                    step="1"
                    value={maxDepth}
                    onChange={(e) => setMaxDepth(Number(e.target.value))}
                    className="slider-input"
                  />
                  <span style={{ fontSize: '0.75rem', color: '#888', marginTop: '4px', display: 'block' }}>
                    Deeper trees capture more patterns but risk overfitting.
                  </span>
                </div>
              ) : (
                /* Linear/Logistic: lr + epochs */
                <>
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
                </>
              )}
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
            🗑️ Clear {model === "linear" ? "Linear" : model === "logistic" ? "Logistic" : model === "knn" ? "KNN" : model === "kmeans" ? "K-Means" : "Decision Tree"} Model Memory
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
            <p>{mode === "train"
              ? (model === "knn" ? "Storing training data for KNN classification..."
                : model === "kmeans" ? "Running K-Means clustering iterations..."
                  : model === "dtree" ? "Building decision tree from training data..."
                    : "Training model weights via gradient descent...")
              : "Executing inference..."}</p>
          </div>
        )}

        {/* ─────────────────────────────────────────────────────────────
         *  TRAINING RESULTS SECTION
         *  Rendered only when: mode === "train" AND trainResults exists
         *  AND we're not currently loading.
         *
         *  Contains model-specific results:
         *    - KNN: sample count, k, dimensions
         *    - K-Means: cluster count, iterations, WCSS curve
         *    - Decision Tree: accuracy, depth, node count
         *    - Linear/Logistic: epochs, final loss, loss curve chart
         * ───────────────────────────────────────────────────────────── */}
        {mode === "train" && trainResults && !loading && (
          <>
            <div className="section-divider" />
            <div className="section-header">
              <h2 className="section-title">📈 Training Results</h2>
              <p className="section-subtitle">
                {model === "knn" ? "Training data stored for KNN classification"
                  : model === "kmeans" ? "K-Means clustering complete"
                    : model === "dtree" ? "Decision tree built successfully"
                      : "Model weights saved successfully"}
              </p>
            </div>

            {model === "knn" ? (
              /* ── KNN results — no loss chart, show data storage metrics ── */
              <div className="metrics-grid">
                <MetricCard value={trainResults.training_samples || "—"} label="Training Samples Stored" />
                <MetricCard value={trainResults.k || k} label="K (Neighbors)" />
                <MetricCard value={trainResults.dimensions || "—"} label="Feature Dimensions" />
                <MetricCard value="KNN" label="Model Type" />
              </div>
            ) : model === "kmeans" ? (
              /* ── K-Means results — WCSS loss curve + cluster metrics ── */
              <>
                <div className="metrics-grid">
                  <MetricCard value={trainResults.k_clusters || kClusters} label="Clusters (K)" />
                  <MetricCard value={trainResults.iterations || "—"} label="Iterations" />
                  <MetricCard
                    value={trainResults.wcss_history ? trainResults.wcss_history[trainResults.wcss_history.length - 1].toFixed(4) : "—"}
                    label="Final WCSS"
                  />
                  <MetricCard value={trainResults.dimensions || "—"} label="Feature Dimensions" />
                </div>
                {trainResults.wcss_history && (
                  <>
                    <div className="section-header">
                      <h2 className="section-title">📉 WCSS Curve (K-Means Convergence)</h2>
                    </div>
                    <div className="chart-card" style={{ height: "400px" }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart
                          data={trainResults.wcss_history.map((w, i) => ({ iteration: i + 1, wcss: w }))}
                          margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                          <XAxis dataKey="iteration" name="Iteration" stroke="#888" label={{ value: 'Iterations', position: 'insideBottom', offset: -10 }} />
                          <YAxis stroke="#888" label={{ value: 'WCSS (Within-Cluster Sum of Squares)', angle: -90, position: 'insideLeft', offset: -10 }} />
                          <Tooltip
                            contentStyle={{
                              background: "#fff",
                              border: "1px solid #eee",
                              borderRadius: "8px",
                              boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
                            }}
                          />
                          <Legend verticalAlign="top" />
                          <Line type="monotone" dataKey="wcss" stroke="#2E86C1" strokeWidth={3} dot={false} activeDot={{ r: 8 }} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </>
                )}
              </>
            ) : model === "dtree" ? (
              /* ── Decision Tree results — accuracy, depth, node count ── */
              <div className="metrics-grid">
                <MetricCard
                  value={trainResults.training_accuracy ? (trainResults.training_accuracy * 100).toFixed(1) + "%" : "—"}
                  label="Training Accuracy"
                />
                <MetricCard value={trainResults.tree_depth || "—"} label="Tree Depth" />
                <MetricCard value={trainResults.node_count || "—"} label="Total Nodes" />
                <MetricCard value="DTREE" label="Model Type" />
              </div>
            ) : (
              /* ── Linear/Logistic results — metric cards + loss curve ── */
              <>
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
          </>
        )}

        {/* ─────────────────────────────────────────────────────────────
         *  PREDICTION RESULTS SECTION
         *  Rendered only when: mode === "predict" AND predictResults exists
         *
         *  Contains:
         *    1. Dynamic accuracy metrics (Classification Accuracy or MAE)
         *    2. A table of all findings with:
         *       - Finding name, severity badge, evidence text
         *       - Actual value and the model's predicted value
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

              {/* ── Dynamic Prediction Accuracy Metrics ──
                  Computes and displays accuracy statistics by comparing the
                  model's predictions against the actual values from the CSV.
                  Three variants are shown depending on the model and target column:
                    - K-Means  → Cluster count summary (unsupervised, no accuracy)
                    - "label"  → Classification Accuracy (% of correct predictions)
                    - "cvss"   → Mean Absolute Error (average prediction offset) */}
              {predictResults.results.length > 0 && (() => {
                /* Check if the CSV contained real label values.
                   If every label is 0, the CSV likely had no label column
                   (the parser defaults missing labels to 0). */
                const hasRealLabels = !predictResults.results.every(f => Number(f.label) === 0);

                /* Same check for CVSS — if every CVSS value is 0, the column
                   was probably absent from the uploaded CSV file. */
                const hasRealCvss = !predictResults.results.every(f => Number(f.cvss) === 0);

                if (model === "kmeans") {
                  /* ── Unsupervised Clustering Summary ──
                     Find the max cluster ID to report the number of clusters generated */
                  const maxCluster = Math.max(...predictResults.results.map(r => Math.round(Number(r[`predicted_${targetCol}`] || 0))));
                  return (
                    <div style={{ background: "#f8fafc", border: "1px solid #e2e8f0", padding: "15px", borderRadius: "8px", margin: "0 20px 20px 20px", color: "#334155" }}>
                      <h3 style={{ margin: 0, fontSize: "1.1rem" }}>📊 Unsupervised Clustering</h3>
                      <p style={{ margin: "5px 0 0 0", fontSize: "0.9rem" }}>Data partitioned into {maxCluster + 1} distinct clusters.</p>
                    </div>
                  );
                }

                if (targetCol === "label" && hasRealLabels) {
                  /* ── Classification Accuracy ──
                     Round each predicted_label to the nearest integer (0 or 1),
                     compare against the actual label, and count how many match.
                     Accuracy = (correct / total) × 100 */
                  let correct = 0;
                  predictResults.results.forEach(f => {
                    let pred = Math.round(Number(f.predicted_label));
                    if (pred === Number(f.label)) correct++;
                  });
                  const acc = ((correct / predictResults.results.length) * 100).toFixed(2);

                  return (
                    <div style={{ background: "#f0fdf4", border: "1px solid #bbf7d0", padding: "15px", borderRadius: "8px", margin: "0 20px 20px 20px", color: "#166534" }}>
                      <h3 style={{ margin: 0, fontSize: "1.1rem" }}>🎯 Classification Accuracy: <strong>{acc}%</strong></h3>
                      <p style={{ margin: "5px 0 0 0", fontSize: "0.9rem" }}>Correctly predicted {correct} out of {predictResults.results.length} samples.</p>
                    </div>
                  );
                }

                if (targetCol === "cvss" && hasRealCvss) {
                  /* ── Mean Absolute Error (MAE) ──
                     For each row, compute |actual_cvss - predicted_cvss|,
                     sum all errors, and divide by the number of samples.
                     Lower MAE = better prediction accuracy. */
                  let totalError = 0;
                  predictResults.results.forEach(f => {
                    totalError += Math.abs(Number(f.cvss) - Number(f.predicted_cvss));
                  });
                  const mae = (totalError / predictResults.results.length).toFixed(3);

                  return (
                    <div style={{ background: "#eff6ff", border: "1px solid #bfdbfe", padding: "15px", borderRadius: "8px", margin: "0 20px 20px 20px", color: "#1e40af" }}>
                      <h3 style={{ margin: 0, fontSize: "1.1rem" }}>📉 Average CVSS Error (MAE): <strong>{mae} points</strong></h3>
                      <p style={{ margin: "5px 0 0 0", fontSize: "0.9rem" }}>On average, the model's CVSS predictions were off by {mae} points.</p>
                    </div>
                  );
                }
              })()}

              {/* Scrollable table container for horizontal overflow on small screens */}
              <div className="findings-table-container">
                <table className="findings-table">
                  <thead>
                    <tr>
                      <th>Finding</th>
                      <th>Severity</th>
                      <th>Evidence</th>
                      {/* Column header changes based on the model and target column */}
                      <th>{targetCol === "cvss" ? "Actual CVSS" : "Actual Label"}</th>
                      <th style={{ color: "#C41E3A" }}>
                        {model === "kmeans" ? "Cluster ID" : `Predicted ${targetCol}`}
                      </th>
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
                        <td>{targetCol === "cvss"
                          ? (predictResults.results.every(r => Number(r.cvss) === 0) ? "-" : f.cvss)
                          : (predictResults.results.every(r => Number(r.label) === 0) ? "-" : f.label)}
                        </td>
                        {/* Show the model's predicted value */}
                        <td style={{ fontWeight: "bold" }}>
                          {model === "kmeans"
                            ? Math.round(Number(f[`predicted_${targetCol}`]))
                            : Number(f[`predicted_${targetCol}`]).toFixed(3)}
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
