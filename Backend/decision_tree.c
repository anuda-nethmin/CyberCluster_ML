/*
Decision Tree Classifier Implementation
Implement a binary classification decision tree from scratch in C.
Uses the Gini impurity criterion for splitting nodes and supports both continuous and categorical features.
Partition the  feature space into pure (or nearly pure) subsets to make predictions.

Algorithm Steps:

1. At each node, scan all features and all possible split points to find the best split based on Gini impurity.
2. recursively split the dataset into subsets based on the best split until a stopping criterion is met (e.g., maximum depth, minimum samples per leaf, or pure node).
3. leaf nodes will store the predicted class label based on the majority class of the samples in that node.

Tree Storage:

nodes are storaged in a dynamic array or linked list structure, where each node contains:
- feature index used for splitting
-left and right child pointers
- predicted class label (for leaf nodes)

GINI Impurity:

Gini(S) = 1 - Σ(pᵢ²)
For binary: Gini = 1 - p₀² - p₁² = 2 * p₀ * p₁
Gini = 0 → perfectly pure node (all same class)
Gini = 0.5 → maximum impurity (50/50 split)
*/

#include "decision_tree.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#define MAX_NODES 4096

//Tree node structure

typedef struct {
    int feature_index;     //Which feature to split on (-1 if leaf)
    double threshold;      // Split threshold value
    int left_child;        // Index of left child node (-1 if leaf)
    int right_child;       // Index of right child node (-1 if leaf)
    int predicted_class;   // Majority class at this node (0 or 1)
    int is_leaf;           // 1 = leaf node, 0 = internal node
} TreeNode;

// Module-level tree storage
static TreeNode tree[MAX_NODES];
static int node_count = 0;
static int tree_depth = 0;

// Calculate Gini impurity for a given set of labels
static double gini_impurity(const int *indices, int count,
                            const double *labels) {
  if (count == 0)
    return 0.0;
  int ones = 0;
  for (int i = 0; i < count; i++) {
    if ((int)labels[indices[i]] == 1)
      ones++;
  }
  double p1 = (double)ones / count;
  double p0 = 1.0 - p1;
  return 1.0 - p0 * p0 - p1 * p1;
}

//Helper Majority class for leaf nodes
static int majority_class(const int *indices, int count, const double *labels) {
  int ones = 0;
  for (int i = 0; i < count; i++) {
    if ((int)labels[indices[i]] == 1)
      ones++;
  }
  return (ones * 2 >= count) ? 1 : 0;
}

// Recursive function to build the decision tree
static int build_tree(const double *features, const double *labels,
                      int *indices, int count, int dim, int depth,
                      int max_depth) {

  if (node_count >= MAX_NODES - 2) {
    // Safety: out of node slots — make a leaf
    int id = node_count++;
    tree[id].is_leaf = 1;
    tree[id].predicted_class = majority_class(indices, count, labels);
    tree[id].left_child = -1;
    tree[id].right_child = -1;
    tree[id].feature_index = -1;
    if (depth > tree_depth)
      tree_depth = depth;
    return id;
  }

  // Track maximum depth reached
  if (depth > tree_depth)
    tree_depth = depth;

  //Stopping conditions
  double current_gini = gini_impurity(indices, count, labels);

  if (depth >= max_depth || count <= 2 || current_gini < 1e-9) {
    int id = node_count++;
    tree[id].is_leaf = 1;
    tree[id].predicted_class = majority_class(indices, count, labels);
    tree[id].left_child = -1;
    tree[id].right_child = -1;
    tree[id].feature_index = -1;
    return id;
  }

  //Find the best split
  int best_feature = -1;
  double best_threshold = 0.0;
  double best_gini = 2.0; /* Worse than any possible Gini */

  /* Temporary arrays for left/right partition */
  int *left_idx = (int *)malloc(count * sizeof(int));
  int *right_idx = (int *)malloc(count * sizeof(int));

  for (int f = 0; f < dim; f++) {
    /* Try each unique value in this feature as a threshold */
    for (int i = 0; i < count; i++) {
      double thresh = features[indices[i] * dim + f];

      int left_count = 0, right_count = 0;
      for (int j = 0; j < count; j++) {
        if (features[indices[j] * dim + f] <= thresh) {
          left_idx[left_count++] = indices[j];
        } else {
          right_idx[right_count++] = indices[j];
        }
      }

      /* Skip trivial splits (everything on one side) */
      if (left_count == 0 || right_count == 0)
        continue;

      /* Weighted Gini of this split */
      double g_left = gini_impurity(left_idx, left_count, labels);
      double g_right = gini_impurity(right_idx, right_count, labels);
      double weighted = ((double)left_count / count) * g_left +
                        ((double)right_count / count) * g_right;

      if (weighted < best_gini) {
        best_gini = weighted;
        best_feature = f;
        best_threshold = thresh;
      }
    }
  }

  /* If no useful split found, make a leaf */
  if (best_feature == -1 || best_gini >= current_gini - 1e-9) {
    free(left_idx);
    free(right_idx);
    int id = node_count++;
    tree[id].is_leaf = 1;
    tree[id].predicted_class = majority_class(indices, count, labels);
    tree[id].left_child = -1;
    tree[id].right_child = -1;
    tree[id].feature_index = -1;
    return id;
  }

    // Perform the best split
  int left_count = 0, right_count = 0;
  for (int j = 0; j < count; j++) {
    if (features[indices[j] * dim + best_feature] <= best_threshold) {
      left_idx[left_count++] = indices[j];
    } else {
      right_idx[right_count++] = indices[j];
    }
  }

  // Allocate this node 
  int id = node_count++;
  tree[id].feature_index = best_feature;
  tree[id].threshold = best_threshold;
  tree[id].is_leaf = 0;
  tree[id].predicted_class = majority_class(indices, count, labels);

  //Recursively build children
  tree[id].left_child = build_tree(features, labels, left_idx, left_count, dim,
                                   depth + 1, max_depth);
  tree[id].right_child = build_tree(features, labels, right_idx, right_count,
                                    dim, depth + 1, max_depth);

  free(left_idx);
  free(right_idx);
  return id;
}

//Helper: Traverse tree for a single prediction
static int classify(const TreeNode *t, const double *point, int dim) {
  int node = 0; /* Start at root */
  (void)dim;    /* dim available for future use */
  while (!t[node].is_leaf) {
    if (point[t[node].feature_index] <= t[node].threshold) {
      node = t[node].left_child;
    } else {
      node = t[node].right_child;
    }
  }
  return t[node].predicted_class;
}