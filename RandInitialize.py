import numpy as np

# Generates a matrix of size (a, b) with random values between -epsilon and epsilon
# a: number of rows = number of neurons in the next layer
# b: number of columns = number of neurons in the current layer + 1 (bias term)
# Each element in the matrix represents the weight for the connection between a neuron in the next layer and a neuron in the current layer.
# The last column in the matrix represents the weights for the bias term for each neuron in the next layer.
def initialise_weights(a, b):
    epsilon = 0.15
    return np.random.rand(a, b) * 2 * epsilon - epsilon
