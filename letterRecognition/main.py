import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from RandInitialize import initialise_weights
from Model import neural_network
from Prediction import predict

def load_csv_data(file_path):
    data = pd.read_csv(file_path, header=None)
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values - 1  # Ensure labels are zero-indexed
    return X, y

# Load the datasets
X_train, y_train = load_csv_data('data/letters/emnist-letters-train.csv')
X_test, y_test = load_csv_data('data/letters/emnist-letters-test.csv')

# Split the train data to get validation data using train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Normalize the input data
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

# Flatten the images
X_train = X_train.reshape(X_train.shape[0], -1)
X_val = X_val.reshape(X_val.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Train the model
input_layer_size = X_train.shape[1]  # Should be 784 for 28x28 pixels
hidden_layer_size = 256
num_labels = 26  # Since there are 26 letters in the EMNIST Letters dataset

# Initialize weights with +1 for the bias term
initial_Theta1 = initialise_weights(hidden_layer_size, input_layer_size + 1)
initial_Theta2 = initialise_weights(hidden_layer_size, hidden_layer_size + 1)
initial_Theta3 = initialise_weights(num_labels, hidden_layer_size + 1)

initial_nn_params = np.concatenate([initial_Theta1.flatten(), initial_Theta2.flatten(), initial_Theta3.flatten()])
max_iter = 500  # Number of iterations for training
regularization_param = 0.01  # Regularization parameter to reduce overfitting
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
np.savetxt('model/letters/Theta1.txt', optimized_Theta1, delimiter=' ')
np.savetxt('model/letters/Theta2.txt', optimized_Theta2, delimiter=' ')
np.savetxt('model/letters/Theta3.txt', optimized_Theta3, delimiter=' ')
