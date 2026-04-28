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
