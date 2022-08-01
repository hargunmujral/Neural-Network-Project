# Creating a function to randomly generate the weights
# and biases of our neural network
# This function takes in the number of neurons in the layer,
# and outputs matrices of random weights and biases

from xml.etree.ElementTree import tostring


def pseudo_random_number(low, high, seed):
    '''Input: accepts a lower and upper bound
    Output: returns a random number'''
    return float(low+(high-low)*(abs(hash(str(hash(str(seed)+str(seed * seed % 3)))))%10**13)/10**13)
        

def random_generation(matrix, seed):
    '''Input: accepts a number of neurons in a layer
    Output: returns matrices of random weights and biases'''
    # Generate a random matrix of weights
    weight_matrix = [[pseudo_random_number(0, 1, seed) for element in row ] for row in matrix]
    bias_matrix = [[pseudo_random_number(0, 1, seed) for element in row ] for row in matrix]
    # Generate a random matrix of biases
    return weight_matrix, bias_matrix
