/*
Logistic Regression  - Binary Classification Engine.
Implementing a gradient descent algorithm to perform multiverse logistic regression.
Used for predicting binary outcomes (0 or 1) based on numerical features, such as classifying vulnerabilities as true or false based on CVSS features.
*/

#include "logistic_regression.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

 /*
 Define the sigmoid function, which maps any real-valued number into the (0, 1) interval.
 This is essential for logistic regression, as it allows us to interpret the output as probabilities.
 */

 static double sigmoid(double z){
    return 1.0 / (1.0 + exp(-z));
 }

 /*
 Function to train logistic regression model using gradient descent optimization.
 minimize the difference (binary cross-entropy loss) between predicted probabilities and actual binary target values.
 */

 int train_logistic_regression(double *features, double *targets, int n, int dim, 
                            double lr, int epochs, const char *weight_file,
                            double *loss_history)
{
    // Guard against empty data sets
    if (n<=0) return -1;

    //Allocating array for weights set to 0.0.
    double *weights = (double *)calloc(dim + 1, sizeof(double));
    if (!weights) return -1;

    // The Gradient descent loop
    for (int e =0; e < epochs; e++)
    {
        // Array to accumulate the slope of the error curve for this weight (nudging directions)
        double *grads = (double*)calloc(dim + 1, sizeof(double));
        double epoch_loss = 0.0;

        // Process each row in the dataset
        for (int i=0; i < n; i++)
        {
            // Linear combination of features and weights
            double z = weights[0]; // bias term
            for (int j=0; j < dim; j++)
            {
                z += weights[j + 1] * features[i * dim + j];
            }

            // Apply sigmoid to get predicted probability
            double y_pred = sigmoid(z);
            double y_actual = targets[i];

            //Prevent log(0) by clamping predicted probabilities
            double y_pred_clipped = y_pred;
            if (y_pred_clipped < 1e-15) y_pred_clipped = 1e-15;
            if (y_pred_clipped > 1 - 1e-15) y_pred_clipped = 1 - 1e-15;

            // Calculate binary cross-entropy loss for this prediction
            //standard formula: -y*log(y_pred) - (1-y)*log(1-y_pred)
            epoch_loss += -y_actual * log(y_pred_clipped) - (1 - y_actual) * log(1 - y_pred_clipped);

            // Calculate gradients for bias and weights
            grads[0] += (y_pred - y_actual); // Gradient for bias
            for (int j=0; j < dim; j++)
            {
                grads[j + 1] += (y_pred - y_actual) * features[i * dim + j]; // Gradient for weights
            }
        }
        // Update weights using the average gradient and learning rate
        for (int j=0; j <=dim; j++)        {
            weights[j] -= lr * (grads[j] / n);
        }
        free(grads); // Free the gradient array

        // Save the loss to the loss array for this epoch
        loss_history[e] = epoch_loss / n;
    }
    // Save the learned weights to a file for later use
    FILE *fp = fopen(weight_file, "wb");
    if (!fp) {
        free(weights);
        return -1;
    }
    // Write dimension and weights to the file
    fwrite(&dim, sizeof(int), 1, fp);
    fwrite(weights, sizeof(double), dim + 1, fp);
    fclose(fp);
    free(weights); // Free the weights array after saving
    return 0;
}

/*
 Function to predict using the trained logistic regression model.
 */
int predict_logistic_regression(double *features, int n, int dim, const char *weight_file, double *predictions){
    FILE *fp = fopen(weight_file, "rb");
    if (!fp) {
        return -1; // Model weights file not found
    }
    int saved_dim;
    if(fread(&saved_dim, sizeof(int), 1, fp) != 1 || saved_dim != dim) {
        fclose(fp);
        return -1; // Dimension mismatch or read error
    }

    double *weights = (double *)malloc((dim + 1) * sizeof(double));
        if (!weights) {
        fclose(fp);
    return -1;
    }

    // Read the weights from the file
    if (fread(weights, sizeof(double), dim + 1, fp) != (size_t)(dim + 1)) {
        free(weights);
        fclose(fp);
        return -1;
    }
    fclose(fp);

    // Generate predictions for each input dataset
    for (int i = 0; i < n; i++) {
        double z = weights[0]; // bias term
        for (int j = 0; j < dim; j++) {
            z += weights[j + 1] * features[i * dim + j];
        }

        // Apply sigmoid to get predicted probability and convert to binary classification (0 or 1) based on a threshold of 0.5
        predictions[i] = sigmoid(z);
        if (predictions[i] > 0.5) {
                predictions[i] = 1;
        } else {
            predictions[i] = 0;
        }
    }
    free(weights); // Free the weights array after prediction
    return 0;

}