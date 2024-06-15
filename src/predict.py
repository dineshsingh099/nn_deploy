import numpy as np
from src.config import config
from src.preprocessing.data_management import load_model

import pipeline as pl

def layer_neurons_weighted_sum(previous_layer_neurons_outputs, current_layer_neurons_biases, current_layer_neurons_weights):
    return current_layer_neurons_biases + np.matmul(previous_layer_neurons_outputs, current_layer_neurons_weights)

def layer_neurons_output(current_layer_neurons_weighted_sums, current_layer_neurons_activation_function):
    if current_layer_neurons_activation_function == "linear":
        return current_layer_neurons_weighted_sums
    elif current_layer_neurons_activation_function == "sigmoid":
        return 1 / (1 + np.exp(-current_layer_neurons_weighted_sums))
    elif current_layer_neurons_activation_function == "tanh":
        return np.tanh(current_layer_neurons_weighted_sums)
    elif current_layer_neurons_activation_function == "relu":
        return np.maximum(current_layer_neurons_weighted_sums, 0)

def predict(input_data):
    h = [None] * config.NUM_LAYERS
    h[0] = input_data

    for l in range(1, config.NUM_LAYERS):
        z = layer_neurons_weighted_sum(h[l-1], pl.theta0[l], pl.theta[l])
        h[l] = layer_neurons_output(z, config.f[l])

    return h[config.NUM_LAYERS-1]

if __name__ == "__main__":
    # Load the model parameters
    model_file = "two_input_xor_nn.pkl"  # Adjust the filename based on your saved model
    loaded_model = load_model(model_file)
    
    # Extract parameters from the loaded model
    pl.theta0 = loaded_model["params"]["biases"]
    pl.theta = loaded_model["params"]["weights"]
    config.f = loaded_model["activations"]

    # Test data
    test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # Make predictions
    predictions = predict(test_data)
    print("Predictions:", predictions)