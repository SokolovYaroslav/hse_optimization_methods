from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from scipy.optimize import line_search, brent, bracket

from logistic_regression.oracle import Oracle


class LineSearch(ABC):
    def __init__(self, oracle: Oracle, max_iter: int, tol: float):
        self._f = oracle
        self._max_iter = max_iter
        self._tol = tol
        self._off_brackets = False

    @abstractmethod
    def __call__(self, w: np.ndarray, direction: np.ndarray) -> float:
        raise NotImplementedError

    @staticmethod
    def get_line_search(
        name: str, oracle: Oracle, max_iter: int = 500, tol: float = 1.4e-8, **kwargs
    ) -> "LineSearch":
        if name == "golden":
            return Golden(oracle, max_iter, tol)
        elif name == "brent":
            return Brent(oracle, max_iter, tol)
        elif name == "armijo":
            return Armijo(oracle, max_iter, tol, **kwargs)
        elif name == "wolf":
            return Wolf(oracle, max_iter, tol, **kwargs)
        elif name == "nesterov":
            return Nesterov(oracle, max_iter, tol, **kwargs)
        else:
            raise ValueError(f"Unknown LineSearch method {name}")

    def turn_off_brackets(self) -> None:
        self._off_brackets = True

    def _bracket(
        self, f: callable, xa: float = 0.0, xb: float = 1.0
    ) -> Tuple[float, float, float]:
        if not self._off_brackets:
            return bracket(f, xa, xb)[:3]
        else:
            return 0.0, 1.0

    def _func(self, w: np.ndarray, direction: np.ndarray) -> callable:
        return lambda alpha: self._f.value(w + alpha * direction)


class Golden(LineSearch):
    def __call__(self, w: np.ndarray, direction: np.ndarray) -> float:
        func = self._func(w, direction)
        return self._golden_search(func, self._bracket(func))

    def _golden_search(self, f: callable, brack: tuple) -> float:
        phi, Phi = (np.sqrt(5) - 1) / 2, 1 - (np.sqrt(5) - 1) / 2
        if len(brack) == 3:
            ax, bx, cx = brack
        else:
            ax, cx = brack
            bx = ax + phi * (cx - ax)

        x0, x3 = ax, cx
        if abs(cx - bx) > abs(bx - ax):
            x1, x2 = bx, bx + Phi * (cx - bx)
        else:
            x2, x1 = bx, bx - Phi * (bx - bx)
        f1, f2 = f(x1), f(x2)

        for _ in range(self._max_iter):
            if abs(x3 - x0) < self._tol * (abs(x1) + abs(x2)):
                break
            if f2 < f1:
                x0, x1, x2 = x1, x2, phi * x2 + Phi * x3
                f1, f2 = f2, f(x2)
            else:
                x3, x2, x1 = x2, x1, phi * x1 + Phi * x0
                f2, f1 = f1, f(x1)

        return x1 if f1 < f2 else x2


class Brent(LineSearch):
    def __call__(self, w: np.ndarray, direction: np.ndarray) -> float:
        func = self._func(w, direction)
        return float(
            brent(
                func, brack=self._bracket(func), tol=self._tol, maxiter=self._max_iter
            )
        )


class Armijo(LineSearch):
    def __init__(self, oracle: Oracle, max_iter: int, tol: float, c: float = 0.5):
        super().__init__(oracle, max_iter, tol)
        self._c = c

    def __call__(self, w: np.ndarray, direction: np.ndarray) -> float:
        func = self._func(w, direction)
        alpha = max(self._bracket(func))

        f_0, df_0 = self._f.fuse_value_grad(w)
        dd = df_0 @ direction
        f_1 = func(alpha)
        for i in range(self._max_iter):
            if f_1 <= f_0 + self._c * alpha * dd:
                break
            alpha *= 0.5
            f_1 = func(alpha)

        return alpha


class Wolf(LineSearch):
    def __init__(
        self,
        oracle: Oracle,
        max_iter: int,
        tol: float,
        c1: float = 1e-4,
        c2: float = 0.9,
    ):
        super().__init__(oracle, max_iter, tol)
        self._c1 = c1
        self._c2 = c2
        self._armijo = Armijo(oracle, max_iter, tol, c1)

    def __call__(self, w: np.ndarray, direction: np.ndarray) -> float:
        res = line_search(
            self._f.value, self._f.grad, w, direction, c1=self._c1, c2=self._c2
        )[0]
        if res is not None:
            return res
        else:
            return self._armijo(w, direction)


class Nesterov(LineSearch):
    def __init__(
        self,
        oracle: Oracle,
        max_iter: int,
        tol: float,
        c1: float = 0.5,
        c2: float = 0.5,
    ):
        super().__init__(oracle, max_iter, tol)
        self._c1 = c1
        self._c2 = c2
        self._alpha = None

    def _init_alpha(self, func):
        if self._alpha is None:
            self._alpha = max(self._bracket(func))

    def __call__(self, w: np.ndarray, direction: np.ndarray) -> float:
        func = self._func(w, direction)
        self._init_alpha(func)

        f_0, f_1 = self._f.value(w), func(self._alpha)
        dd = direction.T @ direction

        for _ in range(self._max_iter):
            if f_1 <= f_0 - self._c1 * self._alpha * dd:
                break
            self._alpha *= self._c2
            f_1 = func(self._alpha)

        self._alpha /= self._c2

        return self._alpha * self._c2
