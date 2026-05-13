/*
json_output.c - JSON serialization for CyberCluster ML backend

PURPOSE:

Contains manula JSON serialization to emit structured direcly from C.
This is essencial bciase Node.js executes this binary and waits for properly formatted stdout output to pass to the react frontend.
Using printf to emit JSON directly avoids the complexity of integrating a full JSON library in C and keeps dependencies minimal.
*/

#include <stdio.h>
#include "json_output.h"

/*
Function to print error
Serializes standard program faults into a JSON scema which the express expects to receive and forward to the frontend for user display.
*/

void print_error(const char *message){
    printf("{\n");
    printf("  \"status\": \"error\",\n");
    printf("  \"message\": \"%s\"\n", message);
    printf("}\n");
}

/*
  Function to print training results
  Serializes the history of the loss curve.
  Iterates through the array and carefully handles comma positioning.
 */
void print_training_results(const double *loss_history, int epochs) {
  printf("{\n");
  printf("  \"status\": \"success\",\n");
  printf("  \"epochs\": %d,\n", epochs);
  printf("  \"loss_history\": [\n");
  
  // Dump the loss values into a Javascript-compatible array format
  for (int i = 0; i < epochs; i++) {
    // If it's the last element, omit the trailing comma to maintain strict JSON compliance
    printf("    %f%s\n", loss_history[i], (i == epochs - 1) ? "" : ",");
  }
  
  printf("  ]\n");
  printf("}\n");
}

/*
  Function to print prediction results
  Fuses the original CSV dataset findings symmetrically with the newly minted AI predictions.
  Outputs a nested list of JSON objects.
 */
void print_prediction_results(const Finding *findings,
                              const double *predictions, int n,
                              const char *target_col) {
  printf("{\n");
  printf("  \"status\": \"success\",\n");
  printf("  \"results\": [\n");
  
  for (int i = 0; i < n; i++) {
    printf("    {\n");
    printf("      \"finding_name\": \"%s\",\n", findings[i].finding_name);
    printf("      \"cvss\": %.1f,\n", findings[i].cvss);
    printf("      \"port\": %d,\n", findings[i].port);
    printf("      \"severity\": \"%s\",\n", findings[i].severity);

    // Note: Escaping massive text blocks natively in C is complex.
    // For this architecture, we safely assume 'evidence' does not contain raw double quotes.
    printf("      \"evidence\": \"%s\",\n", findings[i].evidence);
    printf("      \"label\": %d,\n", findings[i].label);
    
    // Inject dynamic object key string via target_col, typically 'predicted_cvss' or 'predicted_label'
    printf("      \"predicted_%s\": %f\n", target_col, predictions[i]);
    
    // Handle strict JSON trailing comma semantics for the object grouping
    printf("    }%s\n", (i == n - 1) ? "" : ",");
  }
  
  printf("  ]\n");
  printf("}\n");
}

/*
 Function to print kNN training results
 KNN-specific training response. Unlike gradient descent models,
 KNN stores the raw training data rather than learned weights,
 so there is no loss curve to report.
 */
void print_knn_training_results(int n_samples, int dim, int k) {
  printf("{\n");
  printf("  \"status\": \"success\",\n");
  printf("  \"model\": \"knn\",\n");
  printf("  \"k\": %d,\n", k);
  printf("  \"training_samples\": %d,\n", n_samples);
  printf("  \"dimensions\": %d,\n", dim);
  printf("  \"message\": \"Training data stored. Ready for KNN predictions.\"\n");
  printf("}\n");
}

/*
 Function to print kmeans training results
 K-Means training response with WCSS loss curve for chart rendering.
 */
void print_kmeans_training_results(const double *wcss_history, int iterations,
                                   int k, int dim) {
  printf("{\n");
  printf("  \"status\": \"success\",\n");
  printf("  \"model\": \"kmeans\",\n");
  printf("  \"k_clusters\": %d,\n", k);
  printf("  \"iterations\": %d,\n", iterations);
  printf("  \"dimensions\": %d,\n", dim);
  printf("  \"wcss_history\": [\n");
  for (int i = 0; i < iterations; i++) {
    printf("    %f%s\n", wcss_history[i], (i == iterations - 1) ? "" : ",");
  }
  printf("  ]\n");
  printf("}\n");
}

/*
 Function to print dtree training results
 Decision Tree training response with accuracy, depth, and node count.
 */
void print_dtree_training_results(double accuracy, int depth, int nodes) {
  printf("{\n");
  printf("  \"status\": \"success\",\n");
  printf("  \"model\": \"dtree\",\n");
  printf("  \"training_accuracy\": %f,\n", accuracy);
  printf("  \"tree_depth\": %d,\n", depth);
  printf("  \"node_count\": %d,\n", nodes);
  printf("  \"message\": \"Decision tree built. Ready for predictions.\"\n");
  printf("}\n");
}