/*
 * server.cjs — Express.js Proxy Server (CommonJS)
 
  PURPOSE:
    This Node.js server acts as a middleware bridge between the React
    frontend (running on port 5173 via Vite) and the C backend binary
    (ml_engine.exe).  The browser cannot execute native binaries
    directly, so this server:
      1. Receives HTTP requests from the React app
      2. Saves uploaded CSV files to a temporary folder
      3. Spawns ml_engine.exe with the correct command-line arguments
      4. Captures the JSON output from the C program
      5. Sends it back to the React frontend as an HTTP response
 
  ENDPOINTS:
    POST /api/train   — Upload CSV + params → train model → return metrics
    POST /api/predict — Upload CSV + model  → run inference → return results
    POST /api/reset   — Delete saved weights file for a given model
 
  RUN WITH:
    node server.cjs
 
  NOTE: The file uses .cjs extension (CommonJS) because the project's
        package.json has "type": "module", but Express/multer use require().
 */

// Dependencies
const express = require("express");         // Web framework for handling HTTP routes
const multer = require("multer");           // Middleware for parsing multipart/form-data (file uploads)
const cors = require("cors");              // Middleware to allow cross-origin requests (React → Express)
const { execFile } = require("child_process"); // Node built-in to spawn the C binary
const path = require("path");             // Utilities for building file paths
const fs = require("fs");                 // File-system access (read, write, delete files)

// App Initialisation
const app = express();
const PORT = 4000;  // The port this API server listens on

// Enable CORS so the React dev server (port 5173) can call us
// without the browser blocking the requests due to same-origin policy
app.use(cors());

// File Upload Configuration
// Create an "uploads" directory (if it doesn't exist) to temporarily
// store CSV files uploaded by the user before passing them to ml_engine
const uploadDir = path.join(__dirname, "Uploads");
if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir);  // Create the folder on first run
}

// multer stores uploaded files in the uploadDir with auto-generated names and .csv extension
const upload = multer({ 
    dest: uploadDir,
    storage: multer.diskStorage({
        destination: uploadDir,
        filename: (req, file, cb) => {
            cb(null, Date.now() + '.csv');  // Add .csv extension to all uploads
        }
    })
});

// Backend Binary Path
// Resolve the absolute path to the compiled C backend executable.
// The backend directory is assumed to be one level up from this file.
const BACKEND_DIR = path.resolve(__dirname, "..", "backend");
const ML_ENGINE = path.join(BACKEND_DIR, "build", "ml_engine.exe");

// Directory where trained model weight files (.bin) are stored
const modelsDir = path.join(__dirname, "models");
if (!fs.existsSync(modelsDir)) {
    fs.mkdirSync(modelsDir);
}

// ENDPOINT: POST /api/reset
/*
  Deletes the saved weights file for the specified model type.
  This forces the user to retrain the model before making new predictions.
 
  Request body (JSON):
    { model: "linear" | "logistic" | "knn" | "kmeans" | "dtree" }
 
  Response (JSON):
    { status: "success", message: "Cleared saved data for linear model." }
 */
app.post("/api/reset", express.json(), (req, res) => {
    // Default to "linear" if no model type is provided
    const model = req.body.model || "linear";

    // Build the path to the weights file, e.g. models/linear_weights.bin
    const weightsPath = path.join(modelsDir, `${model}_weights.bin`);

    try {
        if (fs.existsSync(weightsPath)) {
            fs.unlinkSync(weightsPath);  // Delete the file
            return res.json({ status: "success", message: `Cleared saved data for ${model} model.` });
        } else {
            // File doesn't exist — no action needed, still report success
            return res.json({ status: "success", message: `No saved data found for ${model} model.` });
        }
    } catch (e) {
        return res.status(500).json({ error: "Failed to clear model data." });
    }
});

// ENDPOINT: POST /api/train
/*
  Accepts a multipart form containing:
    - csv        : the CSV file (uploaded via multer)
    - model      : "linear" or "logistic"
    - target_col : "cvss" or "label"
    - lr         : learning rate (string, e.g. "0.01")
    - epochs     : number of epochs (string, e.g. "1000")
 
  Spawns ml_engine.exe with the "train" sub-command and returns
  the JSON output (loss_history, epochs, etc.) to the React app.
 */
app.post("/api/train", upload.single("csv"), (req, res) => {
    // Guard: ensure a file was actually uploaded
    if (!req.file) return res.status(400).json({ error: "No CSV uploaded" });

    // multer saves the file to disk and gives us the path in req.file.path
    const csvPath = req.file.path;

    // Extract form fields with sensible defaults
    const model = req.body.model || "linear";
    const target_col = req.body.target_col || "cvss";
    const lr = req.body.lr || "0.01";
    const epochs = req.body.epochs || "1000";

    // Path where the C backend will save the trained weight parameters
    const weightsPath = path.join(modelsDir, `${model}_weights.bin`);

    // Command-line arguments for ml_engine.exe:
    //   ml_engine.exe train <csv> <model> <target> <lr> <epochs> <weights_file>
    //   For KNN: lr position carries 'k', epochs is ignored (set to 1)
    let args;
    if (model === "knn") {
        const k = req.body.k || "5";
        args = ["train", csvPath, model, target_col, k, "1", weightsPath];
    } else if (model === "kmeans") {
        // lr position carries k_clusters, epochs position carries max_iter
        args = ["train", csvPath, model, target_col, lr, epochs, weightsPath];
    } else if (model === "dtree") {
        // lr position carries max_depth, epochs is ignored
        args = ["train", csvPath, model, target_col, lr, "1", weightsPath];
    } else {
        args = ["train", csvPath, model, target_col, lr, epochs, weightsPath];
    }

    // Spawn the C binary as a child process with a 60-second timeout and 50MB buffer
    execFile(ML_ENGINE, args, { timeout: 60000, maxBuffer: 1024 * 1024 * 50 }, (error, stdout, stderr) => {
        // Clean up: delete the temporary uploaded CSV regardless of outcome
        try { fs.unlinkSync(csvPath); } catch (e) { }

        // If the process exited with an error, return a 500 error
        if (error) {
            console.error("Train error:", stderr || error.message);
            return res.status(500).json({ error: stderr || error.message });
        }

        // Parse the JSON that the C program printed to stdout
        try {
            const data = JSON.parse(stdout);
            return res.json(data);  // Send the training results to the frontend
        } catch (e) {
            console.error("JSON parse error:", stdout);
            return res.status(500).json({ error: "Failed to parse backend output" });
        }
    });
});

// ENDPOINT: POST /api/predict
/*
  Accepts a multipart form containing:
    - csv        : the CSV file for inference
    - model      : "linear" or "logistic"
    - target_col : "cvss" or "label"
 
  Spawns ml_engine.exe with the "predict" sub-command.  The binary
  reads the previously saved weights file and outputs predictions
  as JSON, which is forwarded to the React frontend.
 */
app.post("/api/predict", upload.single("csv"), (req, res) => {
    if (!req.file) return res.status(400).json({ error: "No CSV uploaded" });

    const csvPath = req.file.path;
    const model = req.body.model || "linear";
    const target_col = req.body.target_col || "cvss";

    // Path to the weights file that was previously saved during training
    const weightsPath = path.join(modelsDir, `${model}_weights.bin`);

    // Command-line arguments:
    //   ml_engine.exe predict <csv> <model> <target> <weights_file> [k]
    const args = ["predict", csvPath, model, target_col, weightsPath];

    // Append k for KNN/K-Means predictions
    if (model === "knn") {
        const k = req.body.k || "5";
        args.push(k);
    } else if (model === "kmeans") {
        const k = req.body.k || "3";
        args.push(k);
    }

    // Spawn with a 30-second timeout and 50MB buffer (inference is faster than training)
    execFile(ML_ENGINE, args, { timeout: 30000, maxBuffer: 1024 * 1024 * 50 }, (error, stdout, stderr) => {
        // Clean up the temporary CSV file
        try { fs.unlinkSync(csvPath); } catch (e) { }

        if (error) {
            console.error("Predict error:", stderr || error.message);
            return res.status(500).json({ error: stderr || error.message });
        }

        try {
            const data = JSON.parse(stdout);
            return res.json(data);  // Send predictions to the frontend
        } catch (e) {
            console.error("JSON parse error:", stdout);
            return res.status(500).json({ error: "Failed to parse backend output" });
        }
    });
});

// Serve Frontend in Production Docker
const distPath = path.join(__dirname, "dist");
if (fs.existsSync(distPath)) {
    app.use(express.static(distPath));
    app.get(/.*/, (req, res) => {
        res.sendFile(path.join(distPath, "index.html"));
    });
}

// Start the Server
app.listen(PORT, () => {
    console.log(`\n  CyberCluster API server running on http://localhost:${PORT}`);
    console.log(`  Backend binary: ${ML_ENGINE}\n`);
});