/*
CVS Parser for the CyberCluster. This module reads a cvs file and extracts the findings parses them to a structured array.
And extracting mathematical vectors to be used for training or inference.
*/

#include "csv_parser.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/*
1. Function for encoding severity
* Maps human readable severity levels to numerical weight.
* This ensures that gradient descent can effectively learn from the severity of findings.
*/

int encode_severity(const char *severity) {
    if (strcmp(severity, "Critical") == 0) {
        return 5;
    } else if (strcmp(severity, "High") == 0) {
        return 4;
    } else if (strcmp(severity, "Medium") == 0) {
        return 3;
    } else if (strcmp(severity, "Low") == 0) {
        return 2;
    } else if (strcmp(severity, "Informational") == 0) {
        return 1;
    } else {
        return 0; // Unknown severity
    }
}

/*
2. Function for decoding severity
* Converts numerical weight back to human readable severity levels.
*/
const char* decode_severity(int weight) {
    switch (weight) {
        case 5: return "Critical";
        case 4: return "High";
        case 3: return "Medium";
        case 2: return "Low";
        case 1: return "Informational";
        default: return "Unknown";
    }
}

/*
3. Function for copying tokens
* Utility function to copy a token from the CSV line and store it in a structured array.
* Forces a null terminator to prevent buffer overflow vulnerabilities.
*/

static void copy_token(char *dest, const char *src, size_t max_len) {
    strncpy(dest, src, max_len - 1);
    dest[max_len - 1] = '\0'; // Ensure null termination
}

/*
4. Function for parsing CSV
* Reads the csv file line by line, extracts the relevant fields, and store them in a structured array (Finding).
*/
int parse_csv(const char *filename, Finding *findings, int max_findings) {
  FILE *fp = fopen(filename, "r");
  if (!fp) {
    // We print purely to stderr rather than stdout, maintaining valid JSON output
    fprintf(stderr, "ERROR: Could not open file: %s\n", filename);
    return -1;
  }
  
  char line[MAX_LINE];
  // Skip the first line automatically (assuming it contains CSV headers)
  if (!fgets(line, sizeof(line), fp)) {
    fprintf(stderr, "ERROR: CSV is empty.\n");
    fclose(fp);
    return -1;
  }
  
  int count = 0;
  
  // Read row-by-row until EOF or until max capacity is reached
  while (fgets(line, sizeof(line), fp) && count < max_findings) {
    // Strip trailing carriage returns and newlines commonly found in Windows/Unix files
    line[strcspn(line, "\r\n")] = 0;
    if (strlen(line) == 0) continue; // Skip blank lines
    
    Finding *f = &findings[count];
    memset(f, 0, sizeof(Finding));   // Zero-out the struct to ensure no lingering memory
    
    // 1. Finding Name
    char *token = strtok(line, ",");
    if (!token) continue;
    copy_token(f->finding_name, token, sizeof(f->finding_name));

    // 2. CVSS Score - convert string to float (atof)
    token = strtok(NULL, ",");
    if (!token) continue;
    f->cvss = (float)atof(token);

    // 3. Port - convert string to integer (atoi)
    token = strtok(NULL, ",");
    if (!token) continue;
    f->port = atoi(token);

    // 4. Severity Tag
    token = strtok(NULL, ",");
    if (!token) continue;
    copy_token(f->severity, token, sizeof(f->severity));

    // 5. Evidence Fragment
    token = strtok(NULL, ",");
    copy_token(f->evidence, token ? token : "", sizeof(f->evidence));

    // 6. Boolean Label
    token = strtok(NULL, ",");
    if (!token) continue;
    f->label = atoi(token);

    count++; // Increment internal row counter
  }
  
  fclose(fp);
  return count; // Return total parsed rowsffdsf
  
}

/*
5. Function for extracting mathematical vectors from the structured array of findings,
*to be used for training or inference.
*/
int extract_data(const Finding *findings, int n, const char *target_col,
                 double **features_out, double **targets_out, int *dim_out) {
  int dim = 0;
  
  /*
  Decide how many input parameters (dimensions) exist based on what we're predicting
  */ 

  if (strcmp(target_col, "cvss") == 0) {
    dim = 3; // Predicting CVSS relies on: Port, Severity Encoding, Evidence Length
  } else if (strcmp(target_col, "label") == 0) {
    dim = 4; // Predicting binary Label relies on: CVSS, Port, Severity Encoding, Evidence Length
  } else {
    return -1; // Unknown configuration 
  }

  *dim_out = dim;

 // Allocate exact memory layout arrays for features [n x dim] and targets [n x 1]
  double *feat = (double *)malloc(n * dim * sizeof(double));
  double *targ = (double *)malloc(n * sizeof(double));
  if (!feat || !targ) {
    if (feat) free(feat);
    if (targ) free(targ);
    return -1;
  }
  
  // Serialize the struct elements dynamically into the arrays
  for (int i = 0; i < n; i++) {
    int base = i * dim;
    if (strcmp(target_col, "cvss") == 0) {
      feat[base + 0] = (double)findings[i].port;
      feat[base + 1] = (double)encode_severity(findings[i].severity);
      feat[base + 2] = (double)strlen(findings[i].evidence);
      targ[i] = (double)findings[i].cvss;
    } else {
      feat[base + 0] = (double)findings[i].cvss;
      feat[base + 1] = (double)findings[i].port;
      feat[base + 2] = (double)encode_severity(findings[i].severity);
      feat[base + 3] = (double)strlen(findings[i].evidence);
      targ[i] = (double)findings[i].label;
    }
  }
  //output of the arrays is passed back via pointer-to-pointer parameters, allowing the caller to receive the allocated memory and use it for ML tasks.
  features_out[0] = feat;
  targets_out[0] = targ;
  return 0;
} 