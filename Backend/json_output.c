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
Print error
Serializes standard program faults into a JSON scema which the express expects to receive and forward to the frontend for user display.
*/

void print_error(const char *message){
    printf("{\n");
    printf("  \"status\": \"error\",\n");
    printf("  \"message\": \"%s\"\n", message);
    printf("}\n");
}