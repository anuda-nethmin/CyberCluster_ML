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
