import time
from abc import ABC, abstractmethod
from typing import Dict

from logistic_regression.line_search import LineSearch
from logistic_regression.oracle import Oracle
import numpy as np


class Opimizer(ABC):
    def __init__(self, oracle: Oracle, line_search_method: str):
        self._f = oracle
        self._line_search = LineSearch.get_line_search(line_search_method, oracle)

        self._times = []
        self._oracul_calls = []
        self._iterations = []
        self._rks = []
        self._timer = None
        self._num_iterations = 0

    def __call__(self, start_point: np.ndarray, tol: float = 1e-8, max_iter: int = 10000) -> np.ndarray:
        self._start_timer()
        return self._optimize(start_point, tol, max_iter)

    def _start_timer(self) -> None:
        self._timer = time.time()

    def _log(self, rk: float) -> None:
        spent_time = time.time() - self._timer
        self._num_iterations += 1

        self._times.append(spent_time)
        self._oracul_calls.append(self._f.num_calls)
        self._iterations.append(self._num_iterations)
        self._rks.append(rk)

    @property
    def stats(self) -> Dict[str, list]:
        return {"times": self._times, "calls": self._oracul_calls, "iters": self._iterations, "errors": self._rks}

    @abstractmethod
    def _optimize(self, start_point: np.ndarray, tol: float = 1e-8, max_iter: int = 10000) -> np.ndarray:
        raise NotImplementedError


class GradientDescent(Opimizer):
    def _optimize(self, start_point: np.ndarray, tol: float = 1e-8, max_iter: int = 10000) -> np.ndarray:
        w, grad = start_point.copy(), self._f.grad(start_point)
        grad_0_norm = grad @ grad

        for _ in range(max_iter):
            alpha = self._line_search(w, -grad)
            w -= alpha * grad

            grad = self._f.grad(w)
            r = (grad @ grad) / grad_0_norm
            self._log(r)
            if r <= tol:
                return w

        return w