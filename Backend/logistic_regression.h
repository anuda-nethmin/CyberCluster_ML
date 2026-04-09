#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

// Function prototype for training logistic regression model
int train_logistic_regression(double *features, double *targets, int n, int dim, 
                            double lr, int epochs, const char *weight_file,
                            double *loss_history);

// Function prototype for predicting using the trained logistic regression model                            
int predict_logistic_regression(double *features, int n, int dim, const char *weight_file, double *predictions);
#endif // LOGISTIC_REGRESSION_H