import math
from typing import Union, Tuple

import numpy as np
from sklearn.datasets import load_svmlight_file
from scipy.sparse import spmatrix

Matrix = Union[np.ndarray, spmatrix]


class Oracle:
    def __init__(self, dataset: Matrix, labels: np.ndarray):
        self._x = dataset
        self._y = labels
        self._calls = 0

    @property
    def num_calls(self) -> int:
        return self._calls

    @staticmethod
    def make_oracle(data_path: str, data_format: str) -> "Oracle":
        assert data_format in ["libsvm", "tsv"]
        if data_format == "tsv":
            data = np.loadtxt(data_path, delimiter="\t")
            x = data[:, 1:]
            y = data[:, :1]
        else:
            x, y = load_svmlight_file(data_path)
            y = (y == 1).astype(np.int8)

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

    def hessian_vec_product(self, w: np.ndarray, d):
        self._calls += 1
        pass

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

    def fuse_value_grad_hessian_vec_product(self, w: np.ndarray, d):
        pass

    @staticmethod
    def _sigmoid(logits: np.ndarray) -> np.ndarray:
        return np.where(
            logits >= 0,
            1 / (1 + np.exp(-logits)),
            np.exp(logits) / (1 + np.exp(logits)),
        )

    @staticmethod
    def _log_loss(probs: np.ndarray, y: np.ndarray) -> float:
        return -np.mean(np.where(y == 1, np.log(probs + 1e-15), np.log(1 - probs + 1e-15)))

    @staticmethod
    def _log_loss_grad(x: Matrix, probs: np.ndarray, y: np.ndarray) -> np.ndarray:
        return ((probs - y) @ x) / x.shape[0]

    @staticmethod
    def _log_loss_hessian(x: Matrix, probs: np.ndarray) -> np.ndarray:
        if isinstance(x, spmatrix):
            return (x.T @ x.multiply((probs * (1 - probs))[:, None])).toarray() / x.shape[0]
        else:
            return (x.T @ (x * (probs * (1 - probs))[:, None])) / x.shape[0]


def test_oracle(path_to_data: str, data_format: str, n_tests: int, seed: int):
    np.random.seed(seed)
    eps = np.sqrt(np.finfo(np.float64).resolution)
    tol = 1e-7
    oracle = Oracle.make_oracle(path_to_data, data_format)
    dim = oracle._x.shape[1]
    for _ in range(n_tests):
        w = np.random.randn(dim)
        _, dloss, d2loss = oracle.fuse_value_grad_hessian(w)
        # Testing gradients
        for i in range(dim):
            h = np.zeros_like(w)
            h[i] = eps
            dloss_num = (oracle.value(w + h) - oracle.value(w - h)) / (2 * eps)
            if not math.isclose(dloss[i], dloss_num, rel_tol=tol, abs_tol=tol):
                print(
                    f"numeric dloss is differ from oracle's dloss: {dloss[i]} and {dloss_num}\n"
                    f"Test: {i}th dim, {path_to_data} data, {seed} seed"
                )
        # Testing Hessian
        for i in range(dim):
            for j in range(dim):
                h = np.zeros_like(w)
                h[j] = eps
                d2loss_num = (oracle.grad(w + h)[i] - oracle.grad(w - h)[i]) / (2 * eps)
                if not math.isclose(float(d2loss[i][j]), d2loss_num, rel_tol=tol, abs_tol=tol):
                    print(
                        f"numeric hessian is differ from oracle's d2loss: {d2loss[i][j]} and {d2loss_num}\n"
                        f"Test: {i}-{j}th dim, {path_to_data} data, {seed} seed"
                    )


def main():
    data_paths = [
        ("data/a1a.txt", "libsvm"),
        ("data/breast-cancer_scale.txt", "libsvm"),
        ("data/generated.tsv", "tsv"),
    ]
    for path_to_data, data_format in data_paths:
        test_oracle(path_to_data, data_format, 1, seed=42)


if __name__ == "__main__":
    main()
