# imports

from Matrix_operations import matrix_dot_prod, matrix_addition, matrix_multiplication
from Loss_function import matrix_mean_squared_error_derivative, matrix_mean_squared_error

# Defining a Gradient Descent Algorithm for Neural Networks
# This function uses the formula: y_pred = w * x + b
# The gradient descent algorithm is used to find the optimal weights and biases
# for a neural network.

# Note that this function may/should be called numerous times

def Single_iter_gradient_descent(input_matrix, output_matrix, weight_matrix, bias_matrix, learning_rate):
    '''Input: accepts an input matrix, an output matrix, a weights matrix, a biases matrix,
     a scalar learning rate, and a scalar number of epochs 
     Output: returns the optimal weights and bias for the neural network'''
    # Calculate the output of the neural network
    y_pred = matrix_multiplication(weight_matrix, input_matrix) + bias_matrix
    # Calculate the loss of the neural network
    loss = matrix_mean_squared_error(output_matrix, y_pred)
    # Calculate the derivative of the loss function
    loss_derivative = matrix_mean_squared_error_derivative(output_matrix, y_pred)
    # Update the weights and biases
    weight_matrix = matrix_addition(matrix_dot_prod(weight_matrix, learning_rate), matrix_multiplication(loss_derivative, input_matrix))
    bias_matrix = matrix_addition(bias_matrix, matrix_dot_prod(loss_derivative, learning_rate))
    return weight_matrix, bias_matrix, loss
