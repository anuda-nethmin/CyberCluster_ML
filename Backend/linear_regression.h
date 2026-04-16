#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

/*
n= number of rows, dim = number of features, features = input data, targets = actual target values, 
lr = learning rate, epochs = number of iterations, loss_history = array to store loss values, weights_file = path to file for saving learned weights.
*/

// Function prototype for training linear regression model
int train_linear_regression(double *features, double *targets, int n, int dim,
                            double lr, int epochs, const char *weight_file,
                            double *loss_history);

// Function prototype for predicting using the linear regression model
int predict_linear_regression(double *features, int n, int dim, const char *weight_file, double *predictions);

#endif // LINEAR_REGRESSION_H