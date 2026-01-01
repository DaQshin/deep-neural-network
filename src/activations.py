import numpy as np

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))

    return A, Z

def relu(Z):
    A = np.maximum(0.0, Z)

    return A, Z

def tanh(Z):
    A = np.tanh(Z)

    return A, Z


def relu_backward(dA, activation_cache):
    Z = activation_cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def tanh_backward(dA, activation_cache):
    Z = activation_cache
    return dA * (1 - np.tanh(Z) ** 2)

def sigmoid_backward(dA, activation_cache):
    Z = activation_cache
    A = 1 / (1 + np.exp(-Z))
    return dA * A * (1 - A)

