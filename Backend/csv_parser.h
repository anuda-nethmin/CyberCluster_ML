/*
Contains the data structures mapping directly to our CyberCluster 
 *   threat-intel schema, as well as the function prototypes required to 
 *   parse CSV datasets into structured memory.
*/
#ifndef CSV_PARSER_H
#define CSV_PARSER_H
#include <stdio.h>
#define MAX_FINDINGS 4096   // Max rows (vulnerabilities) per CSV
#define MAX_LINE 4096 

/*
Creating the core data  structure to hold parsed CSV data, 
mapping directly to our CyberCluster schema.
*/

typedef struct {
  char finding_name[256];  // e.g., "SQL Injection"
  float cvss;              // Continuous target 1 (e.g., 9.8)
  int port;                // Network port (e.g., 80)
  char severity[32];       // categorical label (e.g., "Critical")
  char evidence[512];      // Free-text description or payload
  int label;               // Binary target 2 (1 = exploit, 0 = false positive)
} Finding;

#endif // CSV_PARSER_H

// Function prototype for encoding severity strings to integer codes
int encode_severity(const char *sev);

// Function prototype for decoding severity codes back to strings
const char *decode_severity(int code);

// Function prototype for the CSV parsing function
int parse_csv(const char *filename, Finding *findings, int max_findings);

// Function prototype for the feature/target extraction function
int extract_data(const Finding *findings, int n, const char *target_col, 
  double **features_out, double **targets_out, int *dim_out);