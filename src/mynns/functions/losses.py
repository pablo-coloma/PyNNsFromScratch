import numpy as np


def mse(y_hat: np.ndarray, y: np.ndarray) -> float:
    # SUM / N K
    return float(np.mean((y_hat - y) ** 2))


def bce(y_hat: np.ndarray, y: np.ndarray) -> float:
    eps = 1e-12
    # SUM BINARY ENTROPIES / N
    return float(- np.mean((y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)).sum(axis=1)))


def ce(y_hat: np.ndarray, y: np.ndarray) -> float:
    eps = 1e-12
    # SUM ENTROPIES / N
    return float(- np.mean((y * np.log(y_hat + eps)).sum(axis=1)))
