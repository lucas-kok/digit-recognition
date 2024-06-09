import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from scipy.optimize import minimize
from RandInitialize import initialise_weights
from Model import neural_network
from Prediction import predict

def load_npz_data(file_path):
    data = np.load(file_path)
    return data['x'], data['y']

# Load the datasets
X_train, y_train = load_npz_data('data/shapes/train.npz')
X_val, y_val = load_npz_data('data/shapes/val.npz')
X_test, y_test = load_npz_data('data/shapes/test.npz')

# Normalize the input data
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

# Flatten the images
X_train = X_train.reshape(X_train.shape[0], -1)
X_val = X_val.reshape(X_val.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Train the model
input_layer_size = X_train.shape[1]  # Should be 784 for 28x28 pixels, adjust if needed
hidden_layer_size = 200
num_labels = 4  # Since there are four categories: circle, rectangle, triangle, star

# +1 for the bias term
initial_Theta1 = initialise_weights(hidden_layer_size, input_layer_size + 1)
initial_Theta2 = initialise_weights(hidden_layer_size, hidden_layer_size + 1)
initial_Theta3 = initialise_weights(num_labels, hidden_layer_size + 1)

initial_nn_params = np.concatenate([initial_Theta1.flatten(), initial_Theta2.flatten(), initial_Theta3.flatten()])
max_iter = 500  # Increase this value to train the model for more iterations
regularization_param = 0.3  # Increase this value to reduce overfitting
args = (input_layer_size, hidden_layer_size, num_labels, X_train, y_train, regularization_param)

# Minimize the cost function
# jac=True means that the gradient is provided in the neural_network function
results = minimize(neural_network, x0=initial_nn_params, args=args, options={'maxiter': max_iter, 'disp': True}, method="L-BFGS-B", jac=True) 

optimized_nn_params = results.x

# Reshape the optimized parameters back into the parameters Theta1, Theta2, and Theta3
Theta1_end = hidden_layer_size * (input_layer_size + 1)
Theta2_end = Theta1_end + hidden_layer_size * (hidden_layer_size + 1)

optimized_Theta1 = np.reshape(optimized_nn_params[:Theta1_end], (hidden_layer_size, input_layer_size + 1))
optimized_Theta2 = np.reshape(optimized_nn_params[Theta1_end:Theta2_end], (hidden_layer_size, hidden_layer_size + 1))
optimized_Theta3 = np.reshape(optimized_nn_params[Theta2_end:], (num_labels, hidden_layer_size + 1))

# Test the model
test_predictions = predict(optimized_Theta1, optimized_Theta2, optimized_Theta3, X_test)
print("Test Accuracy: ", np.mean(test_predictions == y_test.flatten()))

train_predictions = predict(optimized_Theta1, optimized_Theta2, optimized_Theta3, X_train)
print("Train Accuracy: ", np.mean(train_predictions == y_train.flatten()))

val_predictions = predict(optimized_Theta1, optimized_Theta2, optimized_Theta3, X_val)
print("Validation Accuracy: ", np.mean(val_predictions == y_val.flatten()))

# Save the model
np.savetxt('model/shapes/Theta1.txt', optimized_Theta1, delimiter=' ')
np.savetxt('model/shapes/Theta2.txt', optimized_Theta2, delimiter=' ')
np.savetxt('model/shapes/Theta3.txt', optimized_Theta3, delimiter=' ')
