/*
main.c Entry point for the CyberCluster ML backend
parses command-line arguments, loads data, and dispatches to the appropriate model training/prediction functions based on user input.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Project headers for model interfaces
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
    if (argc < 4) {
        fprintf(stderr, "Error: Insufficient arguments. Expected at least 4, got %d.\n", argc - 1);
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    printf("Received %d arguments:\n", argc - 1);
    for (int i = 1; i < argc; i++) {
        printf("  arg[%d]: %s\n", i, argv[i]);
    }

    return EXIT_SUCCESS;
}