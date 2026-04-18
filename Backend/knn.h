#ifndef KNN_H
#define KNN_H
#include <stddef.h>

/*
    new_features - pointer to the normalized features for the query point
    all_features - pointer to the flattened [n x dim] training feature matrix
    all_labels   - pointer to the training target array [n]
    n            - number of training samples
    dim          - number of features (dimensions)
    k            - number of neighbors to consider (e.g., 3, 5, 7)

    returns the predicted class label (0 or 1) for the new_features based on majority vote among the k nearest neighbors in the training data
*/

int predict_knn(const double *new_features, const double *all_features,
                const double *all_labels, int n, int dim, int k);

int save_knn_training_data(const double *features, const double *labels,
                           int n, int dim, const char *filepath);

int load_knn_training_data(const char *filepath, double **features_out,
                           double **labels_out, int *n_out, int *dim_out);
                
#endif // KNN_H