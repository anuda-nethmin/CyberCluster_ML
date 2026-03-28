#include <stdio.h>
#include "csv_parser.h"
int main() {
    Finding my_data[10]; // Make a small array of 10 toolboxes
    // Try to run just the parser on the test.csv file
    int rows = parse_csv("test.csv", my_data, 10);
    // Print the results to see if the parser successfully filled the boxes!
    printf("Successfully parsed %d rows!\n", rows);
    if (rows > 0) {
        printf("Row 1 Name: %s\n", my_data[0].finding_name);
        printf("Row 1 CVSS: %f\n", my_data[0].cvss);
    }
    return 0;
}