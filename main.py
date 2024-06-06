from scipy.io import loadmat
from RandInitialize import initialise_weights
import numpy as np
from scipy.optimize import minimize
from Model import neural_network
from Prediction import predict

# Load the dataset
data = loadmat('mnist-original.mat')

# Extract the variables
input_data = data['data'].T
output_labels = data['label'][0]

# Normalize the input data and split it into training and testing data
normalized_input_data = input_data / 255.0

X_train, X_test = normalized_input_data[:60000, :], normalized_input_data[60000:, :]
y_train, y_test = output_labels[:60000], output_labels[60000:]

# Train the model
input_layer_size = 784 # 28x28 pixels
hidden_layer_size = 200
num_labels = 10

# +1 for the bias term
initial_Theta1 = initialise_weights(hidden_layer_size, input_layer_size + 1)
initial_Theta2 = initialise_weights(hidden_layer_size, hidden_layer_size + 1)
initial_Theta3 = initialise_weights(num_labels, hidden_layer_size + 1)

initial_nn_params = np.concatenate([initial_Theta1.flatten(), initial_Theta2.flatten(), initial_Theta3.flatten()])
max_iter = 500 # Increase this value to train the model for more iterations
regularization_param = 0.3 # Increase this value to reduce overfitting
args = (input_layer_size, hidden_layer_size, num_labels, X_train, y_train, regularization_param)

# Minimize the cost function
# jac=True means that the gradient is provided in the neural_network function
results = minimize(neural_network, x0=initial_nn_params, args=args, options={'maxiter': max_iter, 'disp': True}, method="L-BFGS-B", jac=True) 

optimized_nn_params = results.x

print (optimized_nn_params)

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

# Save the model
np.savetxt('model/Theta1.txt', optimized_Theta1, delimiter=' ')
np.savetxt('model/Theta2.txt', optimized_Theta2, delimiter=' ')
np.savetxt('model/Theta3.txt', optimized_Theta3, delimiter=' ')
