#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "csv_parser.h"
#include "linear_regression.h"
#include "logistic_regression.h"
#include "knn.h"
#include "decision_tree.h"

int main(int argc, char *argv[]) {

    /* ─────────────────────────────────────────────
     * 1. LOAD CSV DATA
     * ───────────────────────────────────────────── */
    const char *filename = (argc > 1) ? argv[1] : "synthetic_10k_dataset.csv";
    const int max_rows = 5;

    Finding *my_data = malloc(max_rows * sizeof(*my_data));
    if (!my_data) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    int rows = parse_csv(filename, my_data, max_rows);
    if (rows < 0) {
        printf("Failed to parse %s\n", filename);
        free(my_data);
        return 1;
    }

    double *features = NULL;
    double *targets  = NULL;
    int dim = 0;

    printf("Parsed %d rows\n", rows);

    /* Binary labels */
    double *binary_targets = malloc(rows * sizeof(double));
    for (int i = 0; i < rows; i++)
        binary_targets[i] = (double)my_data[i].label;

    if (extract_data(my_data, rows, "cvss", &features, &targets, &dim) != 0) {
        fprintf(stderr, "extract_data failed\n");
        return 1;
    }

    min_max_normalize_features(features, rows, dim);

    /* ═══════════════════════════════════════
     * LINEAR REGRESSION
     * ═══════════════════════════════════════ */
    printf("\n=== Linear Regression ===\n");

    int epochs = 10000;
    double *loss = malloc(epochs * sizeof(double));

    train_linear_regression(features, targets, rows, dim,
                            0.01, epochs, "models/linear.bin", loss);

    double *pred_lr = malloc(rows * sizeof(double));
    predict_linear_regression(features, rows, dim,
                              "models/linear.bin", pred_lr);

    for (int i = 0; i < rows; i++)
        printf("LR Row %d: Pred=%.3f Actual=%.3f\n",
               i+1, pred_lr[i], targets[i]);

    free(loss);
    free(pred_lr);

    /* ═══════════════════════════════════════
     * LOGISTIC REGRESSION
     * ═══════════════════════════════════════ */
    printf("\n=== Logistic Regression ===\n");

    double *log_loss = malloc(epochs * sizeof(double));

    train_logistic_regression(features, binary_targets, rows, dim,
                              0.01, epochs, "models/logistic.bin", log_loss);

    double *pred_log = malloc(rows * sizeof(double));
    predict_logistic_regression(features, rows, dim,
                                "models/logistic.bin", pred_log);

    for (int i = 0; i < rows; i++) {
        int cls = (pred_log[i] >= 0.5) ? 1 : 0;
        printf("LOG Row %d: Prob=%.3f Class=%d Actual=%d\n",
               i+1, pred_log[i], cls, (int)binary_targets[i]);
    }

    free(log_loss);
    free(pred_log);

    /* ═══════════════════════════════════════
     * KNN
     * ═══════════════════════════════════════ */
    printf("\n=== KNN ===\n");

    save_knn_training_data(features, binary_targets, rows, dim, "models/knn.bin");

    double *knn_feat = NULL;
    double *knn_lbl  = NULL;
    int knn_n, knn_dim;

    load_knn_training_data("models/knn.bin", &knn_feat, &knn_lbl, &knn_n, &knn_dim);

    int correct = 0;
    int k = 3;

    for (int i = 0; i < rows; i++) {
        int pred = predict_knn(&features[i * dim],
                              knn_feat, knn_lbl,
                              knn_n, knn_dim, k);

        if (pred == (int)binary_targets[i]) correct++;

        printf("KNN Row %d: Pred=%d Actual=%d\n",
               i+1, pred, (int)binary_targets[i]);
    }

    printf("KNN Accuracy: %.2f%%\n", 100.0 * correct / rows);

    free(knn_feat);
    free(knn_lbl);

    /* ═══════════════════════════════════════
     * DECISION TREE
     * ═══════════════════════════════════════ */
    printf("\n=== Decision Tree ===\n");

    int max_depth = 5;
    double dt_acc;
    int dt_depth, dt_nodes;

    train_decision_tree(features, binary_targets,
                        rows, dim,
                        max_depth,
                        "models/tree.bin",
                        &dt_acc, &dt_depth, &dt_nodes);

    printf("Tree Depth: %d Nodes: %d Train Acc: %.2f%%\n",
           dt_depth, dt_nodes, dt_acc * 100);

    double *pred_dt = malloc(rows * sizeof(double));

    predict_decision_tree(features, rows, dim,
                          "models/tree.bin", pred_dt);

    int dt_correct = 0;

    for (int i = 0; i < rows; i++) {
        int p = (int)pred_dt[i];
        int a = (int)binary_targets[i];

        if (p == a) dt_correct++;

        printf("DT Row %d: Pred=%d Actual=%d\n", i+1, p, a);
    }

    printf("DT Accuracy: %.2f%%\n", 100.0 * dt_correct / rows);

    free(pred_dt);

    /* Cleanup */
    free(features);
    free(targets);
    free(binary_targets);
    free(my_data);

    return 0;
}