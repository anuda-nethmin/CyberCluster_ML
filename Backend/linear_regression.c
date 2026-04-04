/*
Linear Regression - Mean Squared Error (MSE) Optimization Engine.
Implmenting a gradient decent algorithm to perform multiverse linear regression.
Used for predicting continuous values like CVSS scores (7.5,3.1,9) based on numerical features.
*/

#include "linear_regression.h"
#include <stdio.h>
#include <stdlib.h>

/*
 Function to train linear regression model using gradient descent optimization.
calculate optimal weights to minimize the difference (MSE) between predicted and actual target values.


n= number of rows, dim = number of features, features = input data, targets = actual target values, 
lr = learning rate, epochs = number of iterations, loss_history = array to store loss values, weights_file = path to file for saving learned weights.
*/

int train_linear_regression(double *features, double *targets, int n, int dim, 
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
        // Array to accumulate the slope of the error curvefor this weight (nudgeing directions)
        double *grads = (double*)calloc(dim + 1, sizeof(double));
        double epoch_loss = 0.0;

        // Process each row in the dataset
        for (int i=0; i < n; i++)
        {
            // Y_pred = bias (weight[0]) which is 0.0 at the begining
            double y_pred = weights[0];

            // Calculate the weighted sum of the features for this row
            for (int j=0; j < dim; j++)
            {
                y_pred += weights[j + 1] * features[i * dim + j];
            }

            // Calculate the error for this prediction
            double error = y_pred - targets[i];

            // Add sqared error to epoch loss
            epoch_loss += error * error;

            // Calculate gradients for bias and weights
            grads[0] += error; // Gradient for bias
            for (int j=0; j < dim; j++)
            {
                grads[j + 1] += error * features[i * dim + j]; // Gradient for weights
            }
        }
        // Update weights using the average gradient and learning rate
        for (int j=0; j <=dim; j++)
        {
            weights[j] -= lr * (2.0 / n) * grads[j];
        }
        free(grads); // Free the gradient array

        // Save the MSE to the loss array for this epoch
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

    // Free the weights array after saving
    free(weights);
    return 0;

}

/*
Function to predict target values using the learned weights from the linear regression model.

*/

int predict_linear_regression(double *features, int n, int dim, const char *weight_file, double *predictions)
{
    // Load weights from the file
    FILE *fp = fopen(weight_file, "rb");
    if (!fp) return -1;

    int saved_dim;
    if (fread(&saved_dim, sizeof(int), 1, fp) != 1 || saved_dim != dim) {
        fclose(fp);
        return -1; // Dimension mismatch or read error
    }

    // Allocate exact memory for weights based on the dimension read from the file
    double *weights = (double *)malloc((dim + 1) * sizeof(double));
    if (!weights) {
        fclose(fp);
        return -1;
    }

    // Read the weights from the file
    if (fread(weights, sizeof(double), dim + 1, fp) != (size_t)(dim + 1)) {
        free(weights);
        fclose(fp);
        return -1; // Read error
    }

    // Make predictions using the loaded weights
    for (int i = 0; i < n; i++) {
        double y_pred = weights[0]; // Start with bias
        for (int j = 0; j < dim; j++) {
            y_pred += weights[j + 1] * features[i * dim + j];
        }
        predictions[i] = y_pred;
    }

    free(weights);
    return 0; // Success
}