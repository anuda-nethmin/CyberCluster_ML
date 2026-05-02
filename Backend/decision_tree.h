#ifndef DECISION_TREE_H
#define DECISION_TREE_H

/**
 * train_decision_tree
 *
 * Builds a decision tree classifier on the given labelled dataset.
 *
 * @param features       Flat array [n * dim] of normalised feature values
 * @param labels         Array [n] of binary labels (0 or 1)
 * @param n              Number of training examples
 * @param dim            Number of features per example
 * @param max_depth      Maximum depth of the tree (hyperparameter)
 * @param weights_file   Path to save the tree structure (.bin)
 * @param out_accuracy   Output — training accuracy (0.0 to 1.0)
 * @param out_depth      Output — actual depth of the built tree
 * @param out_nodes      Output — total number of nodes in the tree
 * @return 0 on success, non-zero on failure
 */

// Function to train a decision tree
int train_decision_tree(const double *features, const double *labels, int n, int dim, int max_depth, const char *weights_file, double *out_accuracy, int *out_depth, int *out_nodes);

// Function to predict using a trained decision tree
int predict_decision_tree(const double *features, int n, int dim, const char *weights_file, double *predictions);

#endif // DECISION_TREE_H