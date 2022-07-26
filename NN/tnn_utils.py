import numpy as np

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z

    return A, cache

def relu(Z):
    A = np.maximum(0,Z)
    cache = Z

    return A, cache

def sigmoid_backward(dA, cache):
    
    Z = cache
    dZ = dA * 1/(1+np.exp(-Z)) * (1 - 1/(1+np.exp(-Z)))

    return dZ

def relu_backward(dA, cache):

    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0

    return dZ