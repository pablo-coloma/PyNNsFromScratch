import numpy as np


def sigmoid(t): 
    return 1 / (1 + np.exp(-t))


def d_sigmoid(t):
    s = sigmoid(t)
    return s * (1 - s)


def relu(t): 
    return np.maximum(0, t)


def d_relu(t): 
    arr = np.asarray(t)
    return (arr > 0).astype(arr.dtype)


def tanh(t): 
    return np.tanh(t)


def d_tanh(t): 
    return 1 - np.tanh(t)**2