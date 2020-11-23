from abc import ABC, abstractmethod
import numpy as np

from logistic_regression.oracle import Oracle


class LineSearch(ABC):
    def __init__(self, alpha_0: float = 1):
        self._alpha_0 = alpha_0

    @abstractmethod
    def __call__(self, f: Oracle, w: np.ndarray, direction: np.ndarray) -> float:
        raise NotImplementedError

    @staticmethod
    def get_line_search(name: str, alpha_0: float = 1) -> "LineSearch":
        if name == "golden":
            return Golden(alpha_0)
        elif name == "brent":
            return Brent(alpha_0)
        elif name == "brentd":
            return BrentD(alpha_0)
        elif name == "armijo":
            return Armijo(alpha_0)
        elif name == "wolf":
            return Wolf(alpha_0)
        elif name == "nesterov":
            return Nesterov(alpha_0)
        else:
            raise ValueError(f"Unknown LineSearch method {name}")


class Golden(LineSearch):
    def __call__(self, f: Oracle, w: np.ndarray, direction: np.ndarray) -> float:
        pass


class Brent(LineSearch):
    def __call__(self, f: Oracle, w: np.ndarray, direction: np.ndarray) -> float:
        pass


class BrentD(LineSearch):
    def __call__(self, f: Oracle, w: np.ndarray, direction: np.ndarray) -> float:
        pass


class Armijo(LineSearch):
    def __call__(self, f: Oracle, w: np.ndarray, direction: np.ndarray) -> float:
        pass


class Wolf(LineSearch):
    def __call__(self, f: Oracle, w: np.ndarray, direction: np.ndarray) -> float:
        pass


class Nesterov(LineSearch):
    def __call__(self, f: Oracle, w: np.ndarray, direction: np.ndarray) -> float:
        pass
