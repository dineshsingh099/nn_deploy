import pandas as pd
import numpy as np

from src.config import config
import src.preprocessing.preprocessors as pp
from src.preprocessing.data_management import load_dataset, save_model, load_model

# src/train_pipeline.py
from src import pipeline as pl


z = [None]*config.NUM_LAYERS
h = [None]*config.NUM_LAYERS

del_fl_by_del_z = [None]*config.NUM_LAYERS
del_hl_by_del_theta0 = [None]*config.NUM_LAYERS
del_hl_by_del_theta = [None]*config.NUM_LAYERS
del_L_by_del_h = [None]*config.NUM_LAYERS
del_L_by_del_theta0 = [None]*config.NUM_LAYERS
del_L_by_del_theta = [None]*config.NUM_LAYERS

def layer_neurons_weighted_sum(previous_layer_neurons_outputs, current_layer_neurons_biases, current_layer_neurons_weights):
    return current_layer_neurons_biases + np.matmul(previous_layer_neurons_outputs, current_layer_neurons_weights)

def layer_neurons_output(current_layer_neurons_weighted_sums, current_layer_neurons_activation_function):
    if current_layer_neurons_activation_function == "linear":
        return current_layer_neurons_weighted_sums
    elif current_layer_neurons_activation_function == "sigmoid":
        return 1 / (1 + np.exp(-current_layer_neurons_weighted_sums))
    elif current_layer_neurons_activation_function == "tanh":
        return (np.exp(current_layer_neurons_weighted_sums) - np.exp(-current_layer_neurons_weighted_sums)) / \
               (np.exp(current_layer_neurons_weighted_sums) + np.exp(-current_layer_neurons_weighted_sums))
    elif current_layer_neurons_activation_function == "relu":
        return current_layer_neurons_weighted_sums * (current_layer_neurons_weighted_sums > 0)

def del_layer_neurons_outputs_wrt_weighted_sums(current_layer_neurons_activation_function, current_layer_neurons_weighted_sums):
    if current_layer_neurons_activation_function == "linear":
        return np.ones_like(current_layer_neurons_weighted_sums)
    elif current_layer_neurons_activation_function == "sigmoid":
        current_layer_neurons_outputs = 1 / (1 + np.exp(-current_layer_neurons_weighted_sums))
        return current_layer_neurons_outputs * (1 - current_layer_neurons_outputs)
    elif current_layer_neurons_activation_function == "tanh":
        return (2 / (np.exp(current_layer_neurons_weighted_sums) + np.exp(-current_layer_neurons_weighted_sums)))**2
    elif current_layer_neurons_activation_function == "relu":
        return (current_layer_neurons_weighted_sums > 0)

def del_layer_neurons_outputs_wrt_biases(current_layer_neurons_outputs_dels):
    return current_layer_neurons_outputs_dels

def del_layer_neurons_outputs_wrt_weights(previous_layer_neurons_outputs, current_layer_neurons_outputs_dels):
    return np.matmul(previous_layer_neurons_outputs.T, current_layer_neurons_outputs_dels)

def run_training(tol, epsilon, batch_size=2):
    epoch_counter = 0
    mse = 1
    loss_per_epoch = []
    loss_per_epoch.append(mse)

    training_data = load_dataset("train.csv")

    obj = pp.preprocess_data()
    obj.fit(training_data.iloc[:, 0:2], training_data.iloc[:, 2])
    X_train, Y_train = obj.transform(training_data.iloc[:, 0:2], training_data.iloc[:, 2])

    pl.initialize_parameters()

    while True:
        mse = 0
        num_batches = 0

        for i in range(0, X_train.shape[0], batch_size):
            num_batches += 1
            batch_X = X_train[i:i+batch_size]
            batch_Y = Y_train[i:i+batch_size]

            for j in range(batch_X.shape[0]):
                h[0] = batch_X[j].reshape(1, batch_X.shape[1])

                for l in range(1, config.NUM_LAYERS):
                    z[l] = layer_neurons_weighted_sum(h[l-1], pl.theta0[l], pl.theta[l])
                    h[l] = layer_neurons_output(z[l], config.f[l])

                    del_fl_by_del_z[l] = del_layer_neurons_outputs_wrt_weighted_sums(config.f[l], z[l])
                    del_hl_by_del_theta0[l] = del_layer_neurons_outputs_wrt_biases(del_fl_by_del_z[l])
                    del_hl_by_del_theta[l] = del_layer_neurons_outputs_wrt_weights(h[l-1], del_fl_by_del_z[l])

                L = (1/2) * (batch_Y[j][0] - h[config.NUM_LAYERS-1][0, 0])**2
                mse += L

                del_L_by_del_h[config.NUM_LAYERS-1] = (h[config.NUM_LAYERS-1] - batch_Y[j])
                for l in range(config.NUM_LAYERS-2, 0, -1):
                    del_L_by_del_h[l] = np.matmul(del_L_by_del_h[l+1], (del_fl_by_del_z[l+1] * pl.theta[l+1]).T)

                for l in range(1, config.NUM_LAYERS):
                    del_L_by_del_theta0[l] = del_L_by_del_h[l] * del_hl_by_del_theta0[l]
                    del_L_by_del_theta[l] = del_L_by_del_h[l] * del_hl_by_del_theta[l]

                    pl.theta0[l] = pl.theta0[l] - (epsilon * del_L_by_del_theta0[l])
                    pl.theta[l] = pl.theta[l] - (epsilon * del_L_by_del_theta[l])

        mse /= X_train.shape[0]
        epoch_counter += 1
        loss_per_epoch.append(mse)

        print("Epoch # {}, Loss = {}, Number of Batches = {}".format(epoch_counter, mse, num_batches))

        if abs(loss_per_epoch[epoch_counter] - loss_per_epoch[epoch_counter-1]) < tol:
            break

if __name__ == "__main__":
    run_training(10**(-8), 10**(-7), batch_size=2)
    save_model(pl.theta0, pl.theta)