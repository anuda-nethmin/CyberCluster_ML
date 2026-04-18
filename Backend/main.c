#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "csv_parser.h"
#include "linear_regression.h"
#include "logistic_regression.h"
#include "knn.h"

int main(int argc, char *argv[]) {

    /* ─────────────────────────────────────────────────────────────
     * 1. LOAD CSV DATA
     * ───────────────────────────────────────────────────────────── */

    /* Use command-line filename if provided, otherwise fall back to default */
    const char *filename = (argc > 1) ? argv[1] : "synthetic_10k_dataset.csv";
    const int max_rows = 5;

    /* Allocate array of Finding structs to hold parsed CSV rows */
    Finding *my_data = malloc(max_rows * sizeof(*my_data));
    if (!my_data) {
        fprintf(stderr, "Failed to allocate memory for %d rows\n", max_rows);
        return 1;
    }

    /* Parse the CSV — returns number of rows loaded, or -1 on failure */
    int rows = parse_csv(filename, my_data, max_rows);
    if (rows < 0) {
        printf("Failed to parse %s\n", filename);
        free(my_data);
        return 1;
    }

    /* Feature matrix and target array — filled later by extract_data() */
    double *features = NULL;
    double *targets  = NULL;
    int     dim      = 0;

    /* Print every parsed row so we can verify the CSV loaded correctly */
    printf("Successfully parsed %d rows from %s!\n", rows, filename);
    for (int i = 0; i < rows; ++i) {
        printf("Row %d Name: %s\n",     i + 1, my_data[i].finding_name);
        printf("Row %d CVSS: %f\n",     i + 1, my_data[i].cvss);
        printf("Row %d Port: %d\n",     i + 1, my_data[i].port);
        printf("Row %d Severity: %s\n", i + 1, my_data[i].severity);
        printf("Row %d Evidence: %s\n", i + 1, my_data[i].evidence);
        printf("Row %d Label: %d\n\n",  i + 1, my_data[i].label);
    }

    /* ─────────────────────────────────────────────────────────────
     * 2. PREPARE BINARY TARGETS
     *    Used by logistic regression and KNN (both need 0/1 labels).
     *    Linear regression uses the raw CVSS targets from extract_data.
     * ───────────────────────────────────────────────────────────── */
    double *binary_targets = malloc(rows * sizeof(double));
    if (!binary_targets) {
        fprintf(stderr, "Failed to allocate binary targets\n");
        free(my_data); /* BUG FIX: original code forgot to free my_data here */
        return 1;
    }
    for (int i = 0; i < rows; i++) {
        binary_targets[i] = (double)my_data[i].label; /* 0 or 1 */
    }

    /* ─────────────────────────────────────────────────────────────
     * 3. EXTRACT FEATURE MATRIX
     *    extract_data() builds a flat [rows x dim] feature array
     *    and a [rows] target array from the Finding structs.
     *    "cvss" tells it which field to use as the target.
     * ───────────────────────────────────────────────────────────── */
    int err = extract_data(my_data, rows, "cvss", &features, &targets, &dim);
    if (err != 0) {
        fprintf(stderr, "extract_data failed\n");
        free(my_data);
        free(binary_targets);
        return 1;
    }

    printf("dim = %d\n", dim);

    /* ─────────────────────────────────────────────────────────────
     * 4. NORMALISE FEATURES
     *    Scales every feature column to [0, 1] using min-max scaling.
     *    This is done ONCE and shared by all three models — they all
     *    operate on the same normalised feature matrix.
     * ───────────────────────────────────────────────────────────── */
    min_max_normalize_features(features, rows, dim);

    /* ═════════════════════════════════════════════════════════════
     * MODEL 1 — LINEAR REGRESSION
     * Predicts a continuous value (CVSS score) from the features.
     * ═════════════════════════════════════════════════════════════ */

    printf("\n=== Training Linear Regression on CSV Data ===\n");

    int epochs = 10000;

    /* loss_history stores the MSE at every epoch so we can check convergence */
    double *loss_history = malloc(epochs * sizeof(double));
    if (!loss_history) {
        fprintf(stderr, "Failed to allocate loss history\n");
        free(features); free(targets); free(my_data); free(binary_targets);
        return 1;
    }

    /* Train — learns weights via gradient descent, saves them to .bin file */
    int train_err = train_linear_regression(
        features, targets,    /* normalised X matrix and y targets          */
        rows, dim,            /* dataset dimensions                         */
        0.01,                 /* learning rate — controls step size         */
        epochs,               /* number of gradient descent iterations      */
        "linear_weights.bin", /* where to save the learned weights          */
        loss_history          /* out: MSE recorded at every epoch           */
    );

    if (train_err != 0) {
        fprintf(stderr, "Linear regression training failed\n");
    } else {
        printf("Training complete.\n");
        printf("Loss at epoch 0:    %f\n", loss_history[0]);
        printf("Loss at epoch 5000: %f\n", loss_history[4999]);
        printf("Loss at epoch 9999: %f\n", loss_history[9999]);
    }

    free(loss_history);

    /* ── Linear Regression: Predictions ── */
    printf("\n=== Testing Linear Regression Prediction ===\n");

    /* Allocate one slot per row to hold the predicted CVSS value */
    double *predictions = malloc(rows * sizeof(double));
    if (!predictions) {
        fprintf(stderr, "Failed to allocate predictions array\n");
        free(features); free(targets); free(my_data); free(binary_targets);
        return 1;
    }

    /* Predict — loads saved weights and applies them to the feature matrix */
    int pred_err = predict_linear_regression(
        features,
        rows, dim,
        "linear_weights.bin",
        predictions           /* out: one predicted value per row */
    );

    if (pred_err != 0) {
        fprintf(stderr, "Linear regression prediction failed\n");
        free(predictions); free(features); free(targets);
        free(my_data); free(binary_targets);
        return 1;
    }

    /* Print predicted vs actual with signed error column */
    printf("%-6s %-12s %-12s %-10s\n", "Row", "Predicted", "Actual", "Error");
    printf("----------------------------------------------\n");
    for (int i = 0; i < rows; i++) {
        double error = predictions[i] - targets[i];
        printf("%-6d %-12.4f %-12.4f %-+10.4f\n",
               i + 1, predictions[i], targets[i], error);
    }

    free(predictions);

    /* ═════════════════════════════════════════════════════════════
     * MODEL 2 — LOGISTIC REGRESSION
     * Predicts a probability (0.0–1.0) that a finding is confirmed
     * (label = 1). Threshold at 0.5 to get a hard class prediction.
     * ═════════════════════════════════════════════════════════════ */

    printf("\n=== Training Logistic Regression on CSV Data ===\n");

    int log_epochs = 10000;

    /* Cross-entropy loss recorded at every epoch */
    double *log_loss_history = malloc(log_epochs * sizeof(double));
    if (!log_loss_history) {
        fprintf(stderr, "Failed to allocate logistic loss history\n");
        free(features); free(targets); free(my_data); free(binary_targets);
        return 1;
    }

    /* Train — uses sigmoid activation + binary cross-entropy loss */
    int log_train_err = train_logistic_regression(
        features, binary_targets, /* X matrix and 0/1 label array           */
        rows, dim,
        0.01,                     /* learning rate                           */
        log_epochs,
        "logistic_weights.bin",   /* saved weight file                       */
        log_loss_history          /* out: cross-entropy loss per epoch       */
    );

    if (log_train_err != 0) {
        fprintf(stderr, "Logistic regression training failed\n");
    } else {
        printf("Logistic training complete.\n");
        printf("Loss at epoch 0:    %f\n", log_loss_history[0]);
        printf("Loss at epoch 5000: %f\n", log_loss_history[4999]);
        printf("Loss at epoch 9999: %f\n", log_loss_history[log_epochs - 1]);
    }

    free(log_loss_history);

    /* ── Logistic Regression: Predictions ── */
    printf("\n=== Testing Logistic Regression Prediction ===\n");

    /* Each slot holds a probability in [0, 1] — not a hard class yet */
    double *log_predictions = malloc(rows * sizeof(double));
    if (!log_predictions) {
        fprintf(stderr, "Failed to allocate logistic predictions array\n");
        free(features); free(targets); free(my_data); free(binary_targets);
        return 1;
    }

    int log_pred_err = predict_logistic_regression(
        features,
        rows, dim,
        "logistic_weights.bin",
        log_predictions           /* out: sigmoid probability per row */
    );

    if (log_pred_err != 0) {
        fprintf(stderr, "Logistic regression prediction failed\n");
        free(log_predictions); free(features); free(targets);
        free(my_data); free(binary_targets);
        return 1;
    }

    /* Convert probability to class: >= 0.5 -> 1 (confirmed), else 0 */
    printf("%-6s %-15s %-15s %-12s\n", "Row", "Pred_Prob", "Pred_Class", "Actual");
    printf("----------------------------------------------------------\n");
    for (int i = 0; i < rows; i++) {
        int predicted_class = (log_predictions[i] >= 0.5) ? 1 : 0;
        printf("%-6d %-15.6f %-15d %-12.0f\n",
               i + 1, log_predictions[i], predicted_class, binary_targets[i]);
    }

    free(log_predictions);

    /* ═════════════════════════════════════════════════════════════
     * MODEL 3 — K-NEAREST NEIGHBOURS (KNN)
     *
     * KNN has NO training loop. Instead of learning weights,
     * it stores the entire dataset and classifies new points
     * by majority vote among their k closest neighbours.
     *
     * Steps:
     *   1. Save normalised features + labels to knn_training.bin
     *   2. Load them back (simulates a real train/predict split)
     *   3. For each row, call predict_knn() and compare to actual
     * ═════════════════════════════════════════════════════════════ */

    printf("\n=== Training KNN on CSV Data ===\n");

    /*
     * "Training" = saving the dataset to disk.
     * predict_knn() will search this file at inference time.
     */
    int knn_save_err = save_knn_training_data(
        features,           /* normalised feature matrix [rows x dim] */
        binary_targets,     /* 0/1 label array [rows]                  */
        rows, dim,
        "knn_training.bin"  /* output binary file                      */
    );

    if (knn_save_err != 0) {
        fprintf(stderr, "KNN save training data failed\n");
        free(features); free(targets); free(my_data); free(binary_targets);
        return 1;
    }

    printf("KNN training data saved (%d rows, %d features).\n", rows, dim);

    /* ── KNN: Load Training Data ── */

    double *knn_features = NULL; /* will point to loaded feature matrix */
    double *knn_labels   = NULL; /* will point to loaded label array    */
    int     knn_n        = 0;    /* number of training rows loaded      */
    int     knn_dim      = 0;    /* number of features per row loaded   */

    int knn_load_err = load_knn_training_data(
        "knn_training.bin",
        &knn_features, &knn_labels, /* out: allocated arrays               */
        &knn_n, &knn_dim            /* out: dimensions read from file      */
    );

    if (knn_load_err != 0) {
        fprintf(stderr, "KNN load training data failed\n");
        free(features); free(targets); free(my_data); free(binary_targets);
        return 1;
    }

    printf("KNN training data loaded successfully (%d rows, %d features).\n",
           knn_n, knn_dim);

    /* ── KNN: Predictions ── */
    printf("\n=== Testing KNN Prediction ===\n");

    /*
     * k = 3: consult 3 nearest neighbours per query.
     * Rule of thumb: k = sqrt(n). With 4 rows, 3 is sensible.
     * Use an odd k to avoid ties in the majority vote.
     * When testing on the full 10k dataset, increase to k = 5 or k = 7.
     */
    int k = 3;
    printf("Using k = %d nearest neighbours.\n\n", k);

    printf("%-6s %-15s %-12s\n", "Row", "Pred_Class", "Actual");
    printf("-----------------------------------\n");

    int knn_correct = 0; /* counts how many predictions match the actual label */

    for (int i = 0; i < rows; i++) {
        /*
         * features is a flat [rows x dim] array.
         * Row i starts at index i * dim, so &features[i * dim]
         * gives us a pointer to exactly row i's feature values.
         */
        const double *query = &features[i * dim];

        /*
         * predict_knn():
         *   1. Computes Euclidean distance from query to every training point
         *   2. Sorts all training points by distance (closest first)
         *   3. Counts label=1 and label=0 among the k nearest
         *   4. Returns whichever label has more votes (majority class)
         */
        int predicted_class = predict_knn(
            query,        /* the point we want to classify          */
            knn_features, /* all training points loaded from file   */
            knn_labels,   /* all training labels                    */
            knn_n,        /* total number of training points        */
            knn_dim,      /* number of features per training point  */
            k             /* how many neighbours to consult         */
        );

        int actual_class = (int)binary_targets[i];

        /* Track number of correct predictions for accuracy summary */
        if (predicted_class == actual_class) {
            knn_correct++;
        }

        printf("%-6d %-15d %-12d\n", i + 1, predicted_class, actual_class);
    }

    /*
     * Accuracy = correct predictions / total rows * 100
     * With only 4 rows this number will be 0%, 25%, 50%, 75%, or 100%.
     * More meaningful when run on the full 10k dataset.
     */
    double knn_accuracy = (rows > 0) ? (100.0 * knn_correct / rows) : 0.0;

    printf("-----------------------------------\n");
    printf("KNN Accuracy: %d / %d correct (%.1f%%)\n",
           knn_correct, rows, knn_accuracy);

    /* Free the arrays allocated by load_knn_training_data() */
    free(knn_features);
    free(knn_labels);

    /* ─────────────────────────────────────────────────────────────
     * 5. CLEANUP
     *    Free everything allocated in main before exiting.
     * ───────────────────────────────────────────────────────────── */
    free(features);
    free(targets);
    free(binary_targets);
    free(my_data);

    return 0;
}