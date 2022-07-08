# This file contains the activation functions used in the neural network.

# An activation function is one that defines how a neuron's input is transformed 
# into its output, using weighted sums of the inputs and biases.

# We will define a series of activation functions, each of which will have its
# own unique set of characteristics. The intention is for these functions to be 
# called at will, for experimental and training purposes.

# CONSTANTS
EULERS_NUMBER = 2.718281828459045

# Miscellaneous functions
def exp(x):
    return EULERS_NUMBER ** x

# ReLU
# The ReLU function is simple, taking in an input number and returning 0 if it is
# negative, and the input number if it is positive.

def relu(x):
    return max(0, x)

# Sigmoid
# The sigmoid function takes in an input number, and returns a number between 0 and 1.
# It is an S-shaped curve, with asymptotes at 0 and 1.

def sigmoid(x):
    return 1 / (1 + exp(-x))

# Tanh
# The tanh function returns a number between -1 and 1. It follows a similar
# shape to the sigmoid function, but is more sensitive to the input.

def tanh(x):
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))
