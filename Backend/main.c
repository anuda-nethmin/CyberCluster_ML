/*
main.c Entry point for the CyberCluster ML backend
parses command-line arguments, loads data, and dispatches to the appropriate model training/prediction functions based on user input.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Project headers for model interfaces
#include "csv_parser.h"
#include "json_output.h"
#include "linear_regression.h"
#include "logistic_regression.h"
#include "decision_tree.h"
#include "kmeans.h"
#include "knn.h"

// Print CLI usage instructions
static void print_usage(const char *prog){
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "%s train <csv> <model> <target_col> <lr> <epochs> <weights_file>\n", prog);
    fprintf(stderr, "%s predict <csv> <model> <target_col> <weights_file> [k]\n", prog);
    fprintf(stderr, "Models: linear, logistic, tree, kmeans, knn\n");
}

int main(int argc, char *argv[])  
{
    // Requires at least mode, csv, model
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    // Parses required arguments
    const char *mode = argv[1];
    const char *csv_file = argv[2];
    const char *model = argv[3];

    // Allocate memory for the parsed csv rowes
    Finding *findings = (Finding *)malloc(MAX_FINDINGS * sizeof(Finding));
    if (!findings) {
        print_error("Memory allocation failed for findings");
        return 1;
    }
    // Parse CSV file into the findings array; n = number of valid rows
    int n = parse_csv(csv_file, findings, MAX_FINDINGS);
    if (n <= 0) {
        print_error("No findings parsed from CSV");
        free(findings);
        return 1;
    }

    // Function to handle training mode 
    if (strcmp(mode, "train") == 0) {
        if (argc < 8) {
            print_usage(argv[0]);
            free(findings);
            return 1;
        }
        // Extract training parameters from the command line arguments
        const char *target_col = argv[4];
        double lr = atof(argv[5]); // Learning rate for gradient descent or k, for kNN or max depth for decision tree.
        int epochs = atoi(argv[6]); // Number of training iterations for gradient descent-based models or max iterations for k-means.
        const char *weights_file = argv[7];

        double *features = NULL;
        double *targets = NULL;
        int dim = 0;

        // Convertr findings to feature and target arrays for training
        if (extract_data(findings, n, target_col, &features, &targets, &dim) != 0) {
            print_error("Failed to extract features and targets");
            free(findings);
            return 1;
        }

        // min-max normalize features for better convergence in gradient descent-based models
        min_max_normalize_features(features, n, dim);

        double *loss_history = (double *)malloc(epochs * sizeof(double));
        if (!loss_history) {
            print_error("Memory allocation failed for loss history");
            free(findings);
            free(features);
            free(targets);
            return 1;
        }
        if (strcmp(model, "linear") == 0) {
      if (train_linear_regression(features, targets, n, dim, lr, epochs,
                                  weights_file, loss_history) != 0) {
        print_error("Linear regression training failed");
      } else {
        print_training_results(loss_history, epochs);
      }
    } else if (strcmp(model, "logistic") == 0) {
      if (train_logistic_regression(features, targets, n, dim, lr, epochs,
                                    weights_file, loss_history) != 0) {
        print_error("Logistic regression training failed");
      } else {
        print_training_results(loss_history, epochs);
      }
    } else if (strcmp(model, "knn") == 0) {
      int k = (int)lr;  // k neighbours passed via the lr argument slot
      if (save_knn_training_data(features, targets, n, dim, weights_file) != 0) {
        print_error("KNN data storage failed");
      } else {
        print_knn_training_results(n, dim, k);
      }
    } else if (strcmp(model, "kmeans") == 0) {
      int k_clusters = (int)lr;  // number of clusters passed via lr slot
      if (k_clusters < 2) k_clusters = 2;
      double *wcss_history = (double *)calloc(epochs, sizeof(double));
      int actual_iter = 0;
      if (!wcss_history || train_kmeans(features, n, dim, k_clusters, epochs,
                                        weights_file, wcss_history,
                                        &actual_iter) != 0) {
        print_error("K-Means training failed");
      } else {
        print_kmeans_training_results(wcss_history, actual_iter, k_clusters, dim);
      }
      free(wcss_history);
    } else if (strcmp(model, "dtree") == 0) {
      int max_depth = (int)lr;  // tree depth limit passed via lr slot
      if (max_depth < 1) max_depth = 5;
      double accuracy = 0.0;
      int depth = 0, nodes = 0;
      if (train_decision_tree(features, targets, n, dim, max_depth,
                              weights_file, &accuracy, &depth, &nodes) != 0) {
        print_error("Decision tree training failed");
      } else {
        print_dtree_training_results(accuracy, depth, nodes);
      }
    } else {
      print_error("Unknown model. Use 'linear', 'logistic', 'knn', 'kmeans', or 'dtree'");
    }

    // Free all training memory
    free(loss_history);
    free(features);
    free(targets);

    // Function to handle prediction mode
    } else if (strcmp(mode, "predict") == 0) {
    if (argc < 6) {
      print_usage(argv[0]);
      free(findings);
      return 1;
    }
    const char *target_col = argv[4];
    const char *weights_file = argv[5];

    double *features = NULL;
    double *targets = NULL;
    int dim = 0;

    // Convertr findings to feature and target arrays for prediction
    if (extract_data(findings, n, target_col, &features, &targets, &dim) != 0) {
      print_error("Failed to extract features and targets");
      free(findings);
      return 1;
    }
    
    // min-max normalize features using the same scaling as training (assumes saved min-max values in weights file for simplicity)
    min_max_normalize_features(features, n, dim);

    // Allocate memory for predictions
    double *predictions = (double *)malloc(n * sizeof(double));
    if (!predictions) {
      print_error("Memory allocation failed for predictions");
      free(findings);
      free(features);
      free(targets);
      return 1;
    }

    if (strcmp(model, "linear") == 0) {
      if (predict_linear_regression(features, n, dim, weights_file,
                                    predictions) != 0) {
        print_error("Linear regression prediction failed (weights file missing?)");
      } else {
        print_prediction_results(findings, predictions, n, target_col);
      }
    } else if (strcmp(model, "logistic") == 0) {
      if (predict_logistic_regression(features, n, dim, weights_file,
                                      predictions) != 0) {
        print_error("Logistic regression prediction failed (weights file missing?)");
      } else {
        print_prediction_results(findings, predictions, n, target_col);
      }
    } else if (strcmp(model, "knn") == 0) {
      if (argc < 7) {
        print_error("KNN missing 'k' parameter");
        return 1;
      }
      int k = atoi(argv[6]);
      double *train_features = NULL;
      double *train_labels = NULL;
      int train_n = 0;
      int train_dim = 0;
      if (load_knn_training_data(weights_file, &train_features, &train_labels,
                                 &train_n, &train_dim) != 0) {
        print_error("KNN data load failed (file missing?)");
      } else {
        if (train_dim != dim) {
          print_error("KNN dimensions mismatch");
        } else {
          for (int i = 0; i < n; i++) {
            predictions[i] =
                predict_knn(&features[i * dim], train_features, train_labels,
                            train_n, train_dim, k);
          }
          print_prediction_results(findings, predictions, n, target_col);
        }
        free(train_features);
        free(train_labels);
      }
    } else if (strcmp(model, "kmeans") == 0) {
      // Load saved centroids and assign each row to nearest cluster
      if (predict_kmeans(features, n, dim, weights_file, predictions) != 0) {
        print_error("K-Means prediction failed (centroids file missing?)");
      } else {
        print_prediction_results(findings, predictions, n, target_col);
      }
    } else if (strcmp(model, "dtree") == 0) {
      // Load saved tree structure and classify each row by traversal
      if (predict_decision_tree(features, n, dim, weights_file, predictions) != 0) {
        print_error("Decision tree prediction failed (tree file missing?)");
      } else {
        print_prediction_results(findings, predictions, n, target_col);
      }
    } else {
      print_error("Unknown model. Use 'linear', 'logistic', 'knn', 'kmeans', or 'dtree'");
    }

    // Free all prediction memory
    free(predictions);
    free(features);
    free(targets);
  } else {
    print_error("Unknown mode. Use 'train' or 'predict'");
  }

  // Free findings array and exit
  free(findings);
  return 0;
}