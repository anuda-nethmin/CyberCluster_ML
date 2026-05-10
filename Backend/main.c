/*
main.c Entry point for the CyberCluster ML backend
parses command-line arguments, loads data, and dispatches to the appropriate model training/prediction functions based on user input.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Project headers for model interfaces
#include "csv_parser.h"
#include "json_output.h"
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
    // Requires at least mode, csv, model
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    // Parses required arguments
    const char *mode = argv[1];
    const char *csv_file = argv[2];
    const char *model = argv[3];

    // Allocate memory for the parsed csv rowes
    Finding *findings = (Finding *)malloc(MAX_FINDINGS * sizeof(Finding));
    if (!findings) {
        print_error("Memory allocation failed for findings");
        return 1;
    }
    // Parse CSV file into the findings array; n = number of valid rows
    int n = parse_csv(csv_file, findings, MAX_FINDINGS);
    if (n <= 0) {
        print_error("No findings parsed from CSV");
        free(findings);
        return 1;
    }
    printf("Received %d arguments:\n", argc - 1);
    for (int i = 1; i < argc; i++) {
        printf("  arg[%d]: %s\n", i, argv[i]);
    }
    printf("  Number of findings: %d\n", n);

    // Train mode
}