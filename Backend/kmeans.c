/**
 * kmeans_model.c — K-Means Clustering Implementation
 *
 * PURPOSE:
 *   Implements the K-Means clustering algorithm from scratch in C.
 *   Groups data points into K clusters by iteratively assigning
 *   points to nearest centroids and updating centroid positions.
 *
 * ALGORITHM:
 *   1. Initialise K centroids by picking K random training points
 *   2. REPEAT until convergence or max_iter:
 *      a. Assignment: assign each point to its closest centroid
 *      b. Update: recalculate each centroid as the mean of its members
 *      c. Record WCSS (Within-Cluster Sum of Squares) as the loss
 *   3. Save final centroids to a .bin file
 *
 * LOSS METRIC:
 *   WCSS = Σ (over all points) distance²(point, assigned_centroid)
 *   Lower WCSS = tighter clusters = better fit
 */

#include "kmeans.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

/* Helper: Euclidean distance squared
 * We use squared distance (no sqrt) for comparisons — it's cheaper
 * and the ordering is preserved since sqrt is monotonic.
 * We return the actual distance (with sqrt) for WCSS computation. 
*/
static double euclidean_dist_sq(const double *a, const double *b, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// Helper: Find nearest centroid index
static int nearest_centroid(const double *point, const double *centroids,
                            int k, int dim) {
    int best = 0;
    double best_dist = euclidean_dist_sq(point, &centroids[0], dim);
    for (int c = 1; c < k; c++) {
        double d = euclidean_dist_sq(point, &centroids[c * dim], dim);
        if (d < best_dist) {
            best_dist = d;
            best = c;
        }
    }
    return best;
}

// Function to train kmeans
int train_kmeans(const double *features, int n, int dim, int k,
                 int max_iter, const char *weights_file,
                 double *wcss_history, int *actual_iter) {

    if (k > n) k = n;  // Can't have more clusters than data points

    // Allocate centroids: K vectors of 'dim' dimensions each
    double *centroids = (double *)malloc(k * dim * sizeof(double));
    int *assignments  = (int *)malloc(n * sizeof(int));
    int *counts       = (int *)malloc(k * sizeof(int));
    double *new_centroids = (double *)calloc(k * dim, sizeof(double));

    if (!centroids || !assignments || !counts || !new_centroids) {
        free(centroids); free(assignments); free(counts); free(new_centroids);
        return -1;
    }

    // Step 1: Initialise centroids with K random training points
    srand((unsigned int)time(NULL));
    int *used = (int *)calloc(n, sizeof(int));
    for (int c = 0; c < k; c++) {
        int idx;
        do { idx = rand() % n; } while (used[idx]);
        used[idx] = 1;
        memcpy(&centroids[c * dim], &features[idx * dim], dim * sizeof(double));
    }
    free(used);

    // Step 2: Iterate assignment + update
    int iter;
    for (iter = 0; iter < max_iter; iter++) {

        // 2a. Assignment step: assign each point to nearest centroid and compute WCSS
        double wcss = 0.0;
        for (int i = 0; i < n; i++) {
            assignments[i] = nearest_centroid(&features[i * dim], centroids, k, dim);
            wcss += euclidean_dist_sq(&features[i * dim],
                                      &centroids[assignments[i] * dim], dim);
        }
        wcss_history[iter] = wcss;

        // 2b. Update step: calculate new centroid positions
        memset(new_centroids, 0, k * dim * sizeof(double));
        memset(counts, 0, k * sizeof(int));

        for (int i = 0; i < n; i++) {
            int c = assignments[i];
            counts[c]++;
            for (int j = 0; j < dim; j++) {
                new_centroids[c * dim + j] += features[i * dim + j];
            }
        }

        // Divide sums by counts to get means
        for (int c = 0; c < k; c++) {
            if (counts[c] > 0) {
                for (int j = 0; j < dim; j++) {
                    new_centroids[c * dim + j] /= counts[c];
                }
            }
        }

        // Check for convergence: did centroids change?
        int converged = 1;
        for (int c = 0; c < k * dim; c++) {
            if (fabs(centroids[c] - new_centroids[c]) > 1e-9) {
                converged = 0;
                break;
            }
        }

        // Copy new centroids over old for next iteration
        memcpy(centroids, new_centroids, k * dim * sizeof(double));

        if (converged) {
            iter++;  /* Count this iteration */
            break;
        }
    }

    *actual_iter = iter;

    // Step 3: Save centroids to disk
    FILE *fp = fopen(weights_file, "wb");
    if (!fp) {
        free(centroids); free(assignments); free(counts); free(new_centroids);
        return -1;
    }
    fwrite(&k, sizeof(int), 1, fp);
    fwrite(&dim, sizeof(int), 1, fp);
    fwrite(centroids, sizeof(double), k * dim, fp);
    fclose(fp);

    free(centroids);
    free(assignments);
    free(counts);
    free(new_centroids);
    return 0;
}

// Function to predict kmeans
int predict_kmeans(const double *features, int n, int dim,
                   const char *weights_file, double *predictions) {

    FILE *fp = fopen(weights_file, "rb");
    if (!fp) return -1;

    int saved_k, saved_dim;
    fread(&saved_k, sizeof(int), 1, fp);
    fread(&saved_dim, sizeof(int), 1, fp);

    if (saved_dim != dim) {
        fclose(fp);
        return -1;
    }

    double *centroids = (double *)malloc(saved_k * dim * sizeof(double));
    if (!centroids) { fclose(fp); return -1; }

    fread(centroids, sizeof(double), saved_k * dim, fp);
    fclose(fp);

    // Assign each input row to its nearest centroid
    for (int i = 0; i < n; i++) {
        predictions[i] = (double)nearest_centroid(&features[i * dim],
                                                   centroids, saved_k, dim);
    }

    free(centroids);
    return 0;
}