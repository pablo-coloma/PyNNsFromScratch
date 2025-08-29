import numpy as np


def identity(x: np.ndarray): 
    return x


def d_identity(x: np.ndarray): 
    return np.ones_like(x)


def softmax(x: np.ndarray): 
    # Avoid huge numbers:
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)
