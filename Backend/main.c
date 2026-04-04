#include <stdio.h>
#include <stdlib.h>
#include "csv_parser.h"
#include "linear_regression.h"

int main(int argc, char *argv[]) {
    const char *filename = (argc > 1) ? argv[1] : "synthetic_10k_dataset.csv";
    const int max_rows = 4;
    Finding *my_data = malloc(max_rows * sizeof(*my_data));
    if (!my_data) {
        fprintf(stderr, "Failed to allocate memory for %d rows\n", max_rows);
        return 1;
    }
    // Try to run the parser on the requested CSV file
    int rows = parse_csv(filename, my_data, max_rows);
    if (rows < 0) {
        printf("Failed to parse %s\n", filename);
        free(my_data);
        return 1;
    }

    double *features = NULL;
    double *targets = NULL;
    int dim = 0;

    printf("Successfully parsed %d rows from %s!\n", rows, filename);
    for (int i = 0; i < rows; ++i) {
        printf("Row %d Name: %s\n", i + 1, my_data[i].finding_name);
        printf("Row %d CVSS: %f\n", i + 1, my_data[i].cvss);
        printf("Row %d Port: %d\n", i + 1, my_data[i].port);
        printf("Row %d Severity: %s\n", i + 1, my_data[i].severity);
        printf("Row %d Evidence: %s\n", i + 1, my_data[i].evidence);
        printf("Row %d Label: %d\n\n", i + 1, my_data[i].label);
    }
    int err = extract_data(my_data, rows, "cvss", &features, &targets, &dim);
    if (err != 0) {
        fprintf(stderr, "extract_data failed\n");
    } else {
        printf("dim = %d\n", dim);
        for (int i = 0; i < rows; ++i) {
            printf("row %d: ", i + 1);
            for (int j = 0; j < dim; ++j) {
                printf("%f ", features[i * dim + j]);
            }
            printf(" -> target %f\n", targets[i]);
        }

        printf("\n=== Raw Features (before normalization) ===\n");
        for (int i = 0; i < rows; ++i) {
            printf("row %d: ", i + 1);
            for (int j = 0; j < dim; ++j) {
                printf("%f ", features[i * dim + j]);
            }
            printf(" -> target %f\n", targets[i]);
        }
        printf("\n=== Flat Feature Array (before) ===\n");
        for (int i = 0; i < rows * dim; ++i) {
        printf("[%d] %f\n", i, features[i]);
        }

         // --- Normalize and print ---
        min_max_normalize_features(features, rows, dim);
        printf("\n=== Normalized Features (after normalization) ===\n");
        for (int i = 0; i < rows; ++i) {
            printf("row %d: ", i + 1);
            for (int j = 0; j < dim; ++j) {
                printf("%f ", features[i * dim + j]);
            }
            printf(" -> target %f\n", targets[i]);
        }

        // --- Print flat array after normalization ---
printf("\n=== Flat Feature Array (after) ===\n");
for (int i = 0; i < rows * dim; ++i) {
    printf("[%d] %f\n", i, features[i]);
}
// ---- Train on normalized CSV data ----
        printf("\n=== Training Linear Regression on CSV Data ===\n");
        int epochs = 10000;
        double *loss_history = malloc(epochs * sizeof(double));
        if (!loss_history) {
            fprintf(stderr, "Failed to allocate loss history\n");
            free(features); free(targets); free(my_data);
            return 1;
        }

        int train_err = train_linear_regression(
            features, targets,
            rows, dim,
            0.01,
            epochs,
            "csv_weights.bin",
            loss_history
        );

        if (train_err != 0) {
            fprintf(stderr, "Training failed\n");
        } else {
            printf("Training complete.\n");
            printf("Loss at epoch 0:    %f\n", loss_history[0]);
            printf("Loss at epoch 5000: %f\n", loss_history[4999]);
            printf("Loss at epoch 9999: %f\n", loss_history[9999]);
        }

        free(loss_history);
    // ---- Predict ----
    printf("\n=== Testing Prediction on Training Data ===\n");

    double *predictions = malloc(rows * sizeof(double));
    if (!predictions) {
        fprintf(stderr, "Failed to allocate predictions array\n");
        free(features); free(targets); free(my_data);
        return 1;
    }

    int pred_err = predict_linear_regression(
        features,
        rows, dim,
        "csv_weights.bin",
        predictions
    );

    if (pred_err != 0) {
        fprintf(stderr, "Prediction failed (check csv_weights.bin exists and dim matches)\n");
        free(predictions); free(features); free(targets); free(my_data);
        return 1;
    }

    printf("%-6s %-12s %-12s %-10s\n", "Row", "Predicted", "Actual", "Error");
    printf("----------------------------------------------\n");
    for (int i = 0; i < rows; i++) {
        double error = predictions[i] - targets[i];
        printf("%-6d %-12.4f %-12.4f %-+10.4f\n",
               i + 1, predictions[i], targets[i], error);
    }

    // Summary stats
    double total_error = 0.0;
    for (int i = 0; i < rows; i++) {
        double e = predictions[i] - targets[i];
        total_error += e * e;
    }
    double mse = total_error / rows;
    printf("----------------------------------------------\n");
    printf("MSE on training data: %.6f\n", mse);

    free(predictions);
    free(features);
    free(targets);
    free(my_data);
    return 0;
}
}