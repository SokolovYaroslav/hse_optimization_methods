import functools
from typing import Union, Tuple

import numpy as np
from scipy.special import expit
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from scipy.sparse import spmatrix, hstack, csr_matrix

Matrix = Union[np.ndarray, spmatrix]


class Oracle:
    def __init__(self, dataset: Matrix, labels: np.ndarray):
        self._x = dataset
        self._y = labels
        self._calls = 0

    @property
    def num_calls(self) -> int:
        return self._calls

    def reset_calls(self) -> None:
        self._calls = 0

    @functools.lru_cache()
    def opt(self, tol: float, max_iter: int) -> np.ndarray:
        cls = LogisticRegression(penalty="none", tol=tol, fit_intercept=False, solver="newton-cg", max_iter=max_iter)
        cls.fit(self._x, self._y)
        return cls.coef_[0]

    @property
    def dim(self) -> int:
        return self._x.shape[1]

    @staticmethod
    def make_oracle(data_path: str, data_format: str) -> "Oracle":
        assert data_format in ["libsvm", "tsv"]
        if data_format == "tsv":
            data = np.loadtxt(data_path, delimiter="\t")
            x = np.hstack((data[:, 1:], np.ones((data.shape[0], 1))))
            y = data[:, 0]
        else:
            x, y = load_svmlight_file(data_path)
            x = hstack((x, np.ones((x.shape[0], 1))))
            x = csr_matrix(x)
        classes = np.unique(y)
        assert len(classes) == 2
        y = (y == classes[0]).astype(np.int8)

        return Oracle(x, y)

    def value(self, w: np.ndarray) -> float:
        self._calls += 1
        probs = Oracle._sigmoid(self._x.dot(w))
        return Oracle._log_loss(probs, self._y)

    def grad(self, w: np.ndarray) -> np.ndarray:
        self._calls += 1
        probs = Oracle._sigmoid(self._x.dot(w))
        return Oracle._log_loss_grad(self._x, probs, self._y)

    def hessian(self, w: np.ndarray) -> np.ndarray:
        self._calls += 1
        probs = Oracle._sigmoid(self._x.dot(w))
        return Oracle._log_loss_hessian(self._x, probs)

    def hessian_vec_product(self, w: np.ndarray, d: np.ndarray):
        self._calls += 1
        probs = Oracle._sigmoid(self._x.dot(w))
        return Oracle._log_loss_hessian_vec_prod(self._x, probs, d)

    def fuse_value_grad(self, w: np.ndarray) -> Tuple[float, np.ndarray]:
        self._calls += 1
        probs = Oracle._sigmoid(self._x.dot(w))
        return Oracle._log_loss(probs, self._y), Oracle._log_loss_grad(self._x, probs, self._y)

    def fuse_value_grad_hessian(self, w: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        self._calls += 1
        probs = Oracle._sigmoid(self._x.dot(w))
        return (
            Oracle._log_loss(probs, self._y),
            Oracle._log_loss_grad(self._x, probs, self._y),
            Oracle._log_loss_hessian(self._x, probs),
        )

    def fuse_value_grad_hessian_vec_product(self, w: np.ndarray, d: np.ndarray):
        self._calls += 1
        probs = Oracle._sigmoid(self._x.dot(w))
        return (
            Oracle._log_loss(probs, self._y),
            Oracle._log_loss_grad(self._x, probs, self._y),
            Oracle._log_loss_hessian_vec_prod(self._x, probs, d),
        )

    @staticmethod
    def _sigmoid(logits: np.ndarray) -> np.ndarray:
        return expit(logits)

    @staticmethod
    def _log_loss(probs: np.ndarray, y: np.ndarray) -> float:
        return -np.mean(np.where(y == 1, np.log(probs + 1e-15), np.log(1 - probs + 1e-15)))

    @staticmethod
    def _log_loss_grad(x: Matrix, probs: np.ndarray, y: np.ndarray) -> np.ndarray:
        return ((probs - y) @ x) / x.shape[0]

    @staticmethod
    def _log_loss_hessian(x: Matrix, probs: np.ndarray) -> np.ndarray:
        scale = (probs * (1 - probs))[:, None]
        if isinstance(x, spmatrix):
            return (x.T @ x.multiply(scale)).toarray() / x.shape[0]
        else:
            return (x.T @ (x * scale)) / x.shape[0]

    @staticmethod
    def _log_loss_hessian_vec_prod(x: Matrix, probs: np.ndarray, d: np.ndarray) -> np.ndarray:
        scale = probs * (1 - probs)
        return ((x @ d) * scale) @ x / x.shape[0]


def test_oracle(path_to_data: str, data_format: str, seed: int = 42):
    np.random.seed(seed)
    eps = np.sqrt(np.finfo(np.float64).resolution)
    oracle = Oracle.make_oracle(path_to_data, data_format)
    dim = oracle._x.shape[1]
    w = np.random.randn(dim)
    _, dloss, d2loss = oracle.fuse_value_grad_hessian(w)

    # Testing gradients
    max_error = float("-inf")
    for i in range(dim):
        h = np.zeros_like(w)
        h[i] = eps
        dloss_num = (oracle.value(w + h) - oracle.value(w - h)) / (2 * eps)
        abs_error = abs(dloss[i] - dloss_num)
        max_error = max(max_error, abs_error)
    print(f"Maximum error for gradient is {max_error}")

    # Testing Hessian
    max_error = float("-inf")
    for i in range(dim):
        for j in range(dim):
            h = np.zeros_like(w)
            h[j] = eps
            d2loss_num = (oracle.grad(w + h)[i] - oracle.grad(w - h)[i]) / (2 * eps)
            abs_error = abs(d2loss[i][j] - d2loss_num)
            max_error = max(max_error, abs_error)
    print(f"Maximum error for hessian is {max_error}")


def main():
    data_paths = [
        ("data/a1a.txt", "libsvm"),
        ("data/breast-cancer_scale.txt", "libsvm"),
        ("data/generated.tsv", "tsv"),
    ]
    for path_to_data, data_format in data_paths:
        test_oracle(path_to_data, data_format, seed=42)


if __name__ == "__main__":
    main()
