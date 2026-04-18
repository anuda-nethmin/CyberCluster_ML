/*
KNN - K-Nearest Neighbors algorithm implementation.
Implements the the Euclidean distance logic to calculate the distance between data points and classify them based on the majority class among the nearest neighbors.
*/

#include "knn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Data structure to track distance during sorting

typedef struct {
    double distance;
    int label;
} Neighbor;

//Comparison function for qsort to sort neighbors by distance
static int compare_neighbors(const void *a, const void *b) {
    Neighbor *n1 = (Neighbor *)a;
    Neighbor *n2 = (Neighbor *)b;
    if (n1->distance < n2->distance) return -1;
    if (n1->distance > n2->distance) return 1;
    return 0;
}

/*
Calculate the Euclidean distance between two data points of size of dim.
*/

static double calculate_distance(const double *v1, const double *v2, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim; i++) {
        double diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

//Function to predict the class label for a given input vector using KNN algorithm.

int predict_knn(const double *new_features, const double *all_features,
                const double *all_labels, int n, int dim, int k) {

// 1. Pre-allocate an array to score distance to every trainig point
Neighbor *neighbors = (Neighbor *)malloc(n * sizeof(Neighbor));
if (!neighbors)return -1; // Memory allocation failed

// 2. Compute distance from the new point to every training point
for (int i = 0; i < n; i++) {
    neighbors[i].distance = calculate_distance(new_features, &all_features[i * dim], dim);
    neighbors[i].label = (int)all_labels[i];
}

// 3. Sort neighbors by distance
qsort(neighbors, n, sizeof(Neighbor), compare_neighbors);

// 4. Vote among the k nearest neighbors to determine the predicted class
int count_ones = 0;
int count_zeros = 0;

//look at the k closest neighbors and count their labels
for (int i = 0; i < k; i++) {   
    if (neighbors[i].label == 1) {
        count_ones++;
    } else {
        count_zeros++;
    }
}
free(neighbors); // Clean up allocated memory

// 5. Return the majority class among the neighbors
return (count_ones > count_zeros) ? 1 : 0;

}

/*Function to save training data to a file
writes the full dataset (features and labels) to a binary file for later use in prediction. 
*/

int save_knn_training_data(const double *features, const double *labels,
                           int n, int dim, const char *filepath) {
  FILE *fp = fopen(filepath, "wb");
  if (!fp) return -1;

  // Write the dataset dimensions first
  fwrite(&n, sizeof(int), 1, fp);
  fwrite(&dim, sizeof(int), 1, fp);

  // Write the full flattened feature matrix [n x dim]
  fwrite(features, sizeof(double), n * dim, fp);

  // Write the label array [n]
  fwrite(labels, sizeof(double), n, fp);

  fclose(fp);
  return 0;
}

//Function to load a training dataset from a binary file
int load_knn_training_data(const char *filepath, double **features_out,
                           double **labels_out, int *n_out, int *dim_out) {
  FILE *fp = fopen(filepath, "rb");
  if (!fp) return -1;

  int n, dim;
  // Read dataset dimensions
  if (fread(&n, sizeof(int), 1, fp) != 1 ||
      fread(&dim, sizeof(int), 1, fp) != 1) {
    fclose(fp);
    return -1;
  }

  // Allocate memory for features and labels
  double *features = (double *)malloc(n * dim * sizeof(double));
  double *labels = (double *)malloc(n * sizeof(double));
  if (!features || !labels) {
    if (features) free(features);
    if (labels) free(labels);
    fclose(fp);
    return -1;
  }

  // Read the feature matrix
  if (fread(features, sizeof(double), n * dim, fp) != (size_t)(n * dim)) {
    free(features);
    free(labels);
    fclose(fp);
    return -1;
  }

  // Read the label array
  if (fread(labels, sizeof(double), n, fp) != (size_t)n) {
    free(features);
    free(labels);
    fclose(fp);
    return -1;
  }

  fclose(fp);

  *features_out = features;
  *labels_out = labels;
  *n_out = n;
  *dim_out = dim;
  return 0;
}