#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

// Function prototype for training linear regression model
int train_linear_regression(double *features, double *targets, int n, int dim,
                            double lr, int epochs, const char *weight_file,
                            double *loss_history);

#endif // LINEAR_REGRESSION_H