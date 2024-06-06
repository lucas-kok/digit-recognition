import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

def neural_network(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb):
    # Calculate the number of parameters for Theta1, Theta2, and Theta3
    Theta1_end = hidden_layer_size * (input_layer_size + 1)
    Theta2_end = Theta1_end + hidden_layer_size * (hidden_layer_size + 1)

    # Reshape nn_params back into the parameters Theta1, Theta2, and Theta3
    Theta1 = np.reshape(nn_params[:Theta1_end], (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.reshape(nn_params[Theta1_end:Theta2_end], (hidden_layer_size, hidden_layer_size + 1))
    Theta3 = np.reshape(nn_params[Theta2_end:], (num_labels, hidden_layer_size + 1))

    # Number of training examples
    m = X.shape[0]

    # Add bias unit to the input layer
    X_with_bias = np.hstack((np.ones((m, 1)), X))

    # Forward propagation
    z2 = np.dot(X_with_bias, Theta1.T)
    a2 = sigmoid(z2)
    a2_with_bias = np.hstack((np.ones((a2.shape[0], 1)), a2))

    z3 = np.dot(a2_with_bias, Theta2.T)
    a3 = sigmoid(z3)
    a3_with_bias = np.hstack((np.ones((a3.shape[0], 1)), a3))

    z4 = np.dot(a3_with_bias, Theta3.T)
    a4 = sigmoid(z4)

    # Convert y to one-hot encoding
    y = y.astype(int)
    y_matrix = np.eye(num_labels)[y]

    # Compute the cost
    cost = (-1 / m) * np.sum(y_matrix * np.log(a4) + (1 - y_matrix) * np.log(1 - a4))
    reg = (lamb / (2 * m)) * (np.sum(np.square(Theta1[:, 1:])) + 
                                np.sum(np.square(Theta2[:, 1:])) + 
                                np.sum(np.square(Theta3[:, 1:])))
    cost += reg

    # Backpropagation
    delta4 = a4 - y_matrix
    delta3 = np.dot(delta4, Theta3)[:, 1:] * sigmoid_gradient(z3)
    delta2 = np.dot(delta3, Theta2)[:, 1:] * sigmoid_gradient(z2)

    Delta1 = np.dot(delta2.T, X_with_bias)
    Delta2 = np.dot(delta3.T, a2_with_bias)
    Delta3 = np.dot(delta4.T, a3_with_bias)

    Theta1_grad = Delta1 / m
    Theta2_grad = Delta2 / m
    Theta3_grad = Delta3 / m

    # Regularization for the gradients
    Theta1_grad[:, 1:] += (lamb / m) * Theta1[:, 1:]
    Theta2_grad[:, 1:] += (lamb / m) * Theta2[:, 1:]
    Theta3_grad[:, 1:] += (lamb / m) * Theta3[:, 1:]

    # Unroll gradients into a single vector
    grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel(), Theta3_grad.ravel()])

    return cost, grad