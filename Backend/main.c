#include <stdio.h>
#include <stdlib.h>
#include "csv_parser.h"
int main(int argc, char *argv[]) {
    const char *filename = (argc > 1) ? argv[1] : "synthetic_10k_dataset.csv";
    const int max_rows = 4;
    Finding *my_data = malloc(max_rows * sizeof(*my_data));
    if (!my_data) {
        fprintf(stderr, "Failed to allocate memory for %d rows\n", max_rows);
        return 1;
    }
    // Try to run the parser on the requested CSV file
    int rows = parse_csv(filename, my_data, max_rows);
    if (rows < 0) {
        printf("Failed to parse %s\n", filename);
        free(my_data);
        return 1;
    }

    double *features = NULL;
    double *targets = NULL;
    int dim = 0;

    printf("Successfully parsed %d rows from %s!\n", rows, filename);
    for (int i = 0; i < rows; ++i) {
        printf("Row %d Name: %s\n", i + 1, my_data[i].finding_name);
        printf("Row %d CVSS: %f\n", i + 1, my_data[i].cvss);
        printf("Row %d Port: %d\n", i + 1, my_data[i].port);
        printf("Row %d Severity: %s\n", i + 1, my_data[i].severity);
        printf("Row %d Evidence: %s\n", i + 1, my_data[i].evidence);
        printf("Row %d Label: %d\n\n", i + 1, my_data[i].label);
    }
    int err = extract_data(my_data, rows, "cvss", &features, &targets, &dim);
    if (err != 0) {
        fprintf(stderr, "extract_data failed\n");
    } else {
        printf("dim = %d\n", dim);
        for (int i = 0; i < rows; ++i) {
            printf("row %d: ", i + 1);
            for (int j = 0; j < dim; ++j) {
                printf("%f ", features[i * dim + j]);
            }
            printf(" -> target %f\n", targets[i]);
        }
        free(features);
        free(targets);
    }
    free(my_data);
    return 0;
}