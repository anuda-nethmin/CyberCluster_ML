/**
 * json_output.h — JSON Serialization Interface
 *
 * Exposes functions to translate backend memory structures into 
 * standard stdout textual JSON nodes for frontend consumption.
 */

#ifndef JSON_OUTPUT_H
#define JSON_OUTPUT_H

#include "csv_parser.h"


// Standard training success formatter with loss curve output for gradient descent models
void print_training_results(const double *loss_history, int epochs);

// Standard array prediction formatter
void print_prediction_results(const Finding *findings,
                              const double *predictions, int n,
                              const char *target_col);

/*
 * Used in catch-all failure clauses to pipe standardized 500 error strings 
 * securely back to Express. 
 */
void print_error(const char *message);

/*
 * KNN-specific training output formatter.
 * Unlike linear/logistic, KNN has no loss curve — it stores the raw
 * training data and reports sample count, dimensions, and k value.
 */
void print_knn_training_results(int n_samples, int dim, int k);

/*
 * K-Means training output formatter.
 * Reports WCSS loss curve (for chart), cluster count, iterations, and dimensions.
 */
void print_kmeans_training_results(const double *wcss_history, int iterations,
                                   int k, int dim);

/*
 * Decision Tree training output formatter.
 * Reports training accuracy, tree depth, and total node count.
 */
void print_dtree_training_results(double accuracy, int depth, int nodes);

#endif