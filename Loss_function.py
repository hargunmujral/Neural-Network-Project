# imports

from Matrix_operations import matrix_subtract, matrix_dot_prod

#  Defining a Mean Squared Error Loss Function for Neural Networks
# This function uses the formula: MSE = 1/N * sum( (y_true - y_pred)^2 )

def matrix_mean_squared_error(y_true, y_pred):
    '''Input: accepts two matrices, assumes them to be the same size
    Output: returns the mean squared error of the two matrices'''
    return matrix_dot_prod(sum([(y_true[j][i] - y_pred[j][i]) ** 2 for j in range(len(y_true)) for i in range(len(y_true[0]))]), 2 / len(y_true))


# Defining the derivative of our loss function
# This function uses the formula: dMSE = 2/N * (y_pred - y_true)

def matrix_mean_squared_error_derivative(y_true, y_pred):
    '''Input: accepts two matrices, assumes them to be the same size
    Output: returns the derivative of the mean squared error of the two matrices'''
    return matrix_dot_prod(matrix_subtract(y_pred, y_true), 2 / len(y_true))