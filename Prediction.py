import numpy as np
from Model import sigmoid

def predict(Theta1, Theta2, Theta3, X):
    m = X.shape[0]  # Number of examples

    # Add bias unit to the input layer
    X_with_bias = np.hstack((np.ones((m, 1)), X))

    # Forward propagation to the first hidden layer
    z2 = np.dot(X_with_bias, Theta1.T)
    a2 = sigmoid(z2)

    # Add bias unit to the first hidden layer
    a2_with_bias = np.hstack((np.ones((a2.shape[0], 1)), a2))

    # Forward propagation to the second hidden layer
    z3 = np.dot(a2_with_bias, Theta2.T)
    a3 = sigmoid(z3)

    # Add bias unit to the second hidden layer
    a3_with_bias = np.hstack((np.ones((a3.shape[0], 1)), a3))

    # Forward propagation to the output layer
    z4 = np.dot(a3_with_bias, Theta3.T)
    a4 = sigmoid(z4)

    # Get predictions by finding the index with the highest value in the output layer
    p = np.argmax(a4, axis=1)

    return p
