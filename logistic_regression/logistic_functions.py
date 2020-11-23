from typing import Union

import numpy as np
import scipy

Matrix = Union[np.ndarray, scipy.sparse.spmatrix]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def log_loss(x: Matrix, w: np.ndarray, labels: np.ndarray) -> np.ndarray:
    logits = x.dot(w)
    return -np.mean(np.where(labels == 1, np.log(sigmoid(logits) + 1e-15), -np.log(1 - sigmoid(logits) + 1e-15)))


def log_loss_grad(x: Matrix, w: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return -(labels - sigmoid(x.dot(w))).dot(x) / x.shape[0]


def log_loss_hessian(x: Matrix, w: np.ndarray, labels: np.ndarray) -> np.ndarray:
    probs = sigmoid(x.dot(w))
    return x.T.dot((probs * (1 - probs))).dot(x) / x.shape[0]
