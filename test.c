#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE 4096

typedef struct {
    char finding_name[256];
    float cvss;
    int port;
    char severity[32];
    char evidence[512];
    int label;
} Finding;

// Helper: safely copy a token into a fixed-size char array
void copy_token(char *dest, size_t dest_size, const char *src) {
    if (!src) {
        dest[0] = '\0';
        return;
    }
    strncpy(dest, src, dest_size - 1);
    dest[dest_size - 1] = '\0';
}

int main(int argc, char *argv[]) {

    if (argc < 2) {
        printf("Usage: %s <csv_file>\n", argv[0]);
        printf("Example: %s train.csv\n", argv[0]);
        return 1;
    }

    const char *csv_file = argv[1];

    FILE *fp = fopen(csv_file, "r");
    if (!fp) {
        printf("ERROR: Could not open file: %s\n", csv_file);
        return 1;
    }

    char line[MAX_LINE];

    // Read header line and ignore it
    if (!fgets(line, sizeof(line), fp)) {
        printf("ERROR: CSV is empty.\n");
        fclose(fp);
        return 1;
    }

    int count = 0;

    while (fgets(line, sizeof(line), fp)) {

        // Remove newline
        line[strcspn(line, "\r\n")] = 0;

        // Skip empty lines
        if (strlen(line) == 0) continue;

        Finding f;
        memset(&f, 0, sizeof(Finding));

        // Split the line by commas
        // Expected columns:
        // finding_name,cvss,port,severity,evidence,label
        char *token = strtok(line, ",");
        if (!token) continue;
        copy_token(f.finding_name, sizeof(f.finding_name), token);

        token = strtok(NULL, ",");
        if (!token) continue;
        f.cvss = (float)atof(token);

        token = strtok(NULL, ",");
        if (!token) continue;
        f.port = atoi(token);

        token = strtok(NULL, ",");
        if (!token) continue;
        copy_token(f.severity, sizeof(f.severity), token);

        token = strtok(NULL, ",");
        // evidence can be empty, so allow token to be NULL or empty
        copy_token(f.evidence, sizeof(f.evidence), token ? token : "");

        token = strtok(NULL, ",");
        if (!token) continue;
        f.label = atoi(token);

        count++;

        printf("Finding #%d\n", count);
        printf("  Name     : %s\n", f.finding_name);
        printf("  CVSS     : %.1f\n", f.cvss);
        printf("  Port     : %d\n", f.port);
        printf("  Severity : %s\n", f.severity);
        printf("  Evidence : %s\n", f.evidence);
        printf("  Label    : %d\n", f.label);
        printf("----------------------------------------\n");
    }

    fclose(fp);
    printf("\nDone. Total findings loaded: %d\n", count);

    return 0;
}
