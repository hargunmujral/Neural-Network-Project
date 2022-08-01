# This is the forward propogation algorithm for our neural network.
# This function takes in the input matrix, the output matrix, the weights, 
# the activation function and the biases

from Matrix_operations import matrix_dot_prod

def forward_propogation(Weight_matrix, Bias_matrix, Input_matrix, activation_function):
    '''Input: accepts a weight matrix, a bias matrix, activation function and an input matrix
    Output: returns the output matrix'''
    # Calculate the output of the neural network
    adjustment = matrix_dot_prod(Weight_matrix, Input_matrix) + Bias_matrix
    # apply the sigmoid function to the output matrix
    output_matrix = activation_function(adjustment)
    return output_matrix
