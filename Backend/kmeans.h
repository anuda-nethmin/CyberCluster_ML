/*
 * kmeans_model.h — K-Means Clustering Interface
 *
 * Declares functions for training a K-Means clustering model
 * (unsupervised) and assigning new data points to the nearest
 * learned centroid during prediction.
 *
 * K-Means groups data into K clusters by iteratively:
 *   1. Assigning each point to its nearest centroid (Euclidean distance)
 *   2. Recalculating each centroid as the mean of its assigned points
 *
 * Unlike regression/classification models, K-Means does not use
 * target labels — it discovers structure from features alone.
 */

#ifndef KMEANS_MODEL_H
#define KMEANS_MODEL_H

/**
 * train_kmeans
 *
 * Runs the K-Means algorithm on the given feature matrix.
 *
 * @param features       Flat array [n * dim] of normalised feature values
 * @param n              Number of data points (rows)
 * @param dim            Number of features per point (columns)
 * @param k              Number of clusters to form
 * @param max_iter       Maximum iterations before stopping
 * @param weights_file   Path to save the learned centroids (.bin)
 * @param wcss_history   Output array [max_iter] — WCSS loss per iteration
 * @param actual_iter    Output — how many iterations actually ran
 * @return 0 on success, non-zero on failure
 */
int train_kmeans(const double *features, int n, int dim, int k,
                 int max_iter, const char *weights_file,
                 double *wcss_history, int *actual_iter);

/**
 * predict_kmeans
 *
 * Loads saved centroids and assigns each row to its nearest cluster.
 *
 * @param features       Flat array [n * dim] of normalised feature values
 * @param n              Number of data points
 * @param dim            Number of features per point
 * @param weights_file   Path to the saved centroids (.bin)
 * @param predictions    Output array [n] — cluster ID (0 to k-1) per row
 * @return 0 on success, non-zero on failure
 */
int predict_kmeans(const double *features, int n, int dim,
                   const char *weights_file, double *predictions);

#endif // KMEANS_MODEL_H