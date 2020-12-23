from abc import ABC, abstractmethod
import numpy as np
from sklearn import datasets


class Oracle(ABC):
    def __init__(self, dim: int, name: str):
        assert dim > 0
        self._dim = dim
        self._name = name
        self._eps = None

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self._name

    def f(self, x: np.ndarray) -> float:
        x = self._prepare_x(x)
        f = self._f(x)
        assert isinstance(f, float)
        return f

    def grad(self, x: np.ndarray) -> np.ndarray:
        x = self._prepare_x(x)
        grad = self._grad(x)
        assert grad.shape == (self.dim, 1)
        return grad

    def hess(self, x: np.ndarray) -> np.ndarray:
        x = self._prepare_x(x)
        hess = self._hess(x)
        assert hess.shape == (self.dim, self.dim)
        return hess

    def num_grad(self, x: np.ndarray) -> np.ndarray:
        x = self._prepare_x(x)
        eps = self.eps(x)
        h = np.zeros_like(x)
        grad = np.empty_like(x)
        for i in range(x.shape[0]):
            h[i] = eps
            grad[i] = (self.f(x + h) - self.f(x - h)) / (2 * eps)
            h[i] = 0
        return grad

    def num_hess(self, x: np.ndarray) -> np.ndarray:
        x = self._prepare_x(x)
        eps = self.eps(x)
        h = np.zeros_like(x)
        hess = np.empty_like(x, shape=(x.shape[0], x.shape[0]))
        for i in range(x.shape[0]):
            h[i] = eps
            hess[i] = ((self.grad(x + h) - self.grad(x - h)) / (2 * eps)).flatten()
            h[i] = 0
        return hess

    @abstractmethod
    def _f(self, x: np.ndarray) -> float:
        raise NotImplementedError

    @abstractmethod
    def _grad(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _hess(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _prepare_x(self, x: np.ndarray) -> np.ndarray:
        x = x.reshape(-1, 1)
        assert x.shape == (self.dim, 1)
        return x

    def set_eps(self, eps: float) -> None:
        self._eps = eps

    def eps(self, x: np.ndarray) -> float:
        if self._eps is not None:
            return self._eps
        else:
            return np.sqrt(np.finfo(x.dtype).eps)

    @staticmethod
    def calc_error(true: np.ndarray, pred: np.ndarray) -> float:
        true_norm = np.linalg.norm(true)
        pred_norm = np.linalg.norm(pred)
        rel_error = (true_norm - pred_norm) / true_norm
        return rel_error


class OracleDummy(Oracle):
    def __init__(self, dim: int):
        assert dim == 1
        super().__init__(1, "dummy")

    def _f(self, x: np.ndarray) -> float:
        return float(np.power(x, 3))

    def _grad(self, x: np.ndarray) -> np.ndarray:
        return 3 * np.power(x, 2)

    def _hess(self, x: np.ndarray) -> np.ndarray:
        return 6 * x


class Oracle_41(Oracle):
    def __init__(self, dim: int):
        super().__init__(dim, "4.1")
        self._A = datasets.make_spd_matrix(self.dim)

    def _f(self, x: np.ndarray) -> float:
        return np.linalg.norm(x @ x.T - self._A, ord="fro") ** 2 / 2

    def _grad(self, x: np.ndarray) -> np.ndarray:
        return 2 * (x @ x.T - self._A) @ x

    def _hess(self, x: np.ndarray) -> np.ndarray:
        E = np.eye(x.shape[0], dtype=x.dtype)
        return 2 * ((x.T @ x) * E + 2 * x @ x.T - self._A)


class Oracle_42(Oracle):
    def __init__(self, dim: int):
        super().__init__(dim, "4.2")
        self._A = datasets.make_spd_matrix(self.dim)

    def _f(self, x: np.ndarray) -> float:
        return float(((self._A @ x).T @ x) / (x.T @ x))

    def _grad(self, x: np.ndarray) -> np.ndarray:
        x_2 = x.T @ x
        return (self._A @ x * x_2 - (x.T @ self._A @ x @ x.T).T) * 2 / (x_2 ** 2)

    def _hess(self, x: np.ndarray) -> np.ndarray:
        E = np.eye(x.shape[0], dtype=x.dtype)
        x_2 = x.T @ x

        res = x_2 * self._A
        res -= 2 * self._A @ x @ x.T
        res -= (self._A @ x).T @ x * E
        res -= 2 * x @ (self._A @ x).T
        res += (self._A @ x).T @ x * (x @ x.T) * 4 / x_2
        res *= 2 / (x_2 ** 2)
        return res


class Oracle_43(Oracle):
    def __init__(self, dim: int):
        super().__init__(dim, "4.3")
        self._A = datasets.make_spd_matrix(self.dim)

    def _f(self, x: np.ndarray) -> float:
        x_2 = x.T @ x
        return float(np.power(x_2, x_2))

    def _grad(self, x: np.ndarray) -> np.ndarray:
        x_2 = x.T @ x
        return 2 * np.power(x_2, x_2) * (np.log(x_2) + 1) * x

    def _hess(self, x: np.ndarray) -> np.ndarray:
        E = np.eye(x.shape[0], dtype=x.dtype)
        x_2 = x.T @ x
        res = 2 * (np.log(x_2) + 1) ** 2
        res += 1 / (x.T @ x)
        res = res * x @ x.T
        res += (np.log(x_2) + 1) * E
        res *= 2 * np.power(x_2, x_2)
        return res


class Oracle_61(Oracle):
    def __init__(self, dim: int):
        assert dim == 2
        super().__init__(2, "6.1")

    def _f(self, x: np.ndarray) -> float:
        x, y = x[0], x[1]
        return float(2 * x ** 2 + y ** 2 * (x ** 2 - 2))

    def _grad(self, x: np.ndarray) -> np.ndarray:
        x, y = x[0], x[1]
        x_grad = 4 * x + 2 * y ** 2 * x
        y_grad = 2 * x ** 2 * y - 4 * y
        return np.array([x_grad, y_grad]).reshape(2, 1)

    def _hess(self, x: np.ndarray) -> np.ndarray:
        x, y = x[0], x[1]
        xx_grad = 4 + 2 * y ** 2
        xy_grad = 4 * x * y
        yy_grad = 2 * x ** 2 - 4
        return np.array([xx_grad, xy_grad, xy_grad, yy_grad]).reshape(2, 2)


class Oracle_62(Oracle):
    def __init__(self, dim: int):
        assert dim == 2
        super().__init__(dim, "6.2")
        self._lambda = 5

    def _f(self, x: np.ndarray) -> float:
        x, y = x[0], x[1]
        return float((1 - x) ** 2 + self._lambda * (y - x ** 2) ** 2)

    def _grad(self, x: np.ndarray) -> np.ndarray:
        x, y = x[0], x[1]
        x_grad = -2 + 2 * x - 4 * self._lambda * (y * x - x ** 3)
        y_grad = 2 * self._lambda * (y - x ** 2)
        return np.array([x_grad, y_grad]).reshape(2, 1)

    def _hess(self, x: np.ndarray) -> np.ndarray:
        x, y = x[0], x[1]
        xx_grad = float(2 + 12 * self._lambda * x ** 2 - 4 * self._lambda * y)
        xy_grad = float(-4 * self._lambda * x)
        yy_grad = 2 * self._lambda
        return np.array([xx_grad, xy_grad, xy_grad, yy_grad]).reshape(2, 2)


def test():
    n = 50
    oracles = [OracleDummy(1), Oracle_41(n), Oracle_42(n), Oracle_43(n), Oracle_61(2), Oracle_62(2)]
    for oracle in oracles:
        x = np.random.uniform(size=(oracle.dim,))
        grad_error = Oracle.calc_error(oracle.grad(x), oracle.num_grad(x))
        hess_error = Oracle.calc_error(oracle.hess(x), oracle.num_hess(x))
        print(f"{oracle.name} | grad max rel error: {grad_error} | hess max rel error: {hess_error}")


if __name__ == '__main__':
    test()
