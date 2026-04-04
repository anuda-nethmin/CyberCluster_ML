#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

// Function prototype for training linear regression model
int train_linear_regression(double *features, double *targets, int n, int dim,
                            double lr, int epochs, const char *weight_file,
                            double *loss_history);

// Function prototype for predicting using the linear regression model
int predict_linear_regression(double *features, int n, int dim, const char *weight_file, double *predictions);

#endif // LINEAR_REGRESSION_H