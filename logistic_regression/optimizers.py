import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, Tuple, Optional, Deque

from scipy.linalg import cho_factor, LinAlgError, cho_solve, svd

from logistic_regression.line_search import LineSearch
from logistic_regression.oracle import Oracle
import numpy as np


class Optimizer(ABC):
    def __init__(
        self, oracle: Oracle, line_search: LineSearch, start_point: np.ndarray, tol: float = 1e-8, max_iter: int = 10000
    ):
        self._oracle = oracle
        self._line_search = line_search
        self._start_point = start_point
        self._tol = tol
        self._max_iter = max_iter

        self._times = []
        self._oracle_calls = []
        self._iterations = []
        self._grad_norms = []
        self._loss_diffs = []
        self._timer = None
        self._num_iterations = 0
        self._opt_loss = oracle.value(oracle.opt(tol, max_iter))

    @staticmethod
    def get_optimizer(
        name: str,
        oracle: Oracle,
        line_search: LineSearch,
        start_point: np.ndarray,
        tol: float = 1e-8,
        max_iter: int = 10000,
        history_size: Optional[int] = None,
    ) -> "Optimizer":
        if name == "GD":
            assert history_size is None, f"history_size is an argument only for l-bfgs optimizer"
            return GradientDescent(oracle, line_search, start_point, tol, max_iter)

        elif name == "newton-chol":
            assert history_size is None, f"history_size is an argument only for l-bfgs optimizer"
            line_search.turn_off_brackets()
            return NewtonCholecky(oracle, line_search, start_point, tol, max_iter)

        elif name == "newton-svd":
            assert history_size is None, f"history_size is an argument only for l-bfgs optimizer"
            line_search.turn_off_brackets()
            return NewtonSVD(oracle, line_search, start_point, tol, max_iter)

        elif name == "hessian-free":
            assert history_size is None, f"history_size is an argument only for l-bfgs optimizer"
            line_search.turn_off_brackets()
            return HessianFree(oracle, line_search, start_point, tol, max_iter)

        elif name == "l-bfgs":
            assert history_size is not None, f"You must provide a history_size, got {history_size}"
            assert history_size > 0, f"You must provide a positive history_size, got {history_size}"
            line_search.turn_off_brackets()
            return LBFGS(oracle, line_search, start_point, history_size, tol, max_iter)

        else:
            raise ValueError(f"Unknown optimizer {name}")

    def __call__(self) -> np.ndarray:
        self._start_timer()
        w = self._start_point.copy()

        loss, grad = self._call_to_oracle(w)
        grad_0_norm = grad @ grad

        for _ in range(self._max_iter):
            direction = self._get_direction(grad)
            if np.linalg.norm(direction) > 1e2:
                direction = direction / np.linalg.norm(direction) * 1e2
            alpha = self._line_search(w, direction)
            w = w + alpha * direction

            loss, grad = self._call_to_oracle(w)

            grad_norm = (grad @ grad) / grad_0_norm
            loss_diff = abs(loss - self._opt_loss)
            self._log(loss_diff, grad_norm)
            if grad_norm <= self._tol:
                break
        return w

    def _start_timer(self) -> None:
        self._oracle.reset_calls()
        self._timer = time.time()

    def _log(self, loss_diff: float, grad_norm: float) -> None:
        spent_time = time.time() - self._timer
        self._num_iterations += 1

        self._times.append(spent_time)
        self._oracle_calls.append(self._oracle.num_calls)
        self._iterations.append(self._num_iterations)
        self._loss_diffs.append(np.log10(loss_diff))
        self._grad_norms.append(np.log10(grad_norm))

    @property
    def stats(self) -> Dict[str, list]:
        return {
            "times": self._times,
            "calls": self._oracle_calls,
            "iters": self._iterations,
            "grad_norm": self._grad_norms,
            "loss_diffs": self._loss_diffs,
        }

    @abstractmethod
    def _call_to_oracle(self, w: np.ndarray) -> Tuple[float, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def _get_direction(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplemented


class GradientDescent(Optimizer):
    def _call_to_oracle(self, w: np.ndarray) -> Tuple[float, np.ndarray]:
        return self._oracle.fuse_value_grad(w)

    def _get_direction(self, grad: np.ndarray) -> np.ndarray:
        return -grad


class NewtonCholecky(Optimizer):
    def __init__(
        self, oracle: Oracle, line_search: LineSearch, start_point: np.ndarray, tol: float = 1e-8, max_iter: int = 10000
    ):
        super().__init__(oracle, line_search, start_point, tol, max_iter)
        self._tau = 1e-8

    def _call_to_oracle(self, w: np.ndarray) -> Tuple[float, np.ndarray]:
        val, grad, hessian = self._oracle.fuse_value_grad_hessian(w)
        self._hessian = hessian
        return val, grad

    def _get_direction(self, grad: np.ndarray) -> np.ndarray:
        tau_has_changed = False
        while True:
            try:
                factor = cho_factor(self._hessian)
                break
            except LinAlgError:
                dim = self._hessian.shape[0]
                self._hessian[range(dim), range(dim)] += self._tau
                self._tau *= 2
                tau_has_changed = True
        if tau_has_changed:
            self._tau /= 2
        return cho_solve(factor, -grad)


class NewtonSVD(Optimizer):
    def _call_to_oracle(self, w: np.ndarray) -> Tuple[float, np.ndarray]:
        val, grad, hessian = self._oracle.fuse_value_grad_hessian(w)
        self._hessian = hessian
        return val, grad

    def _get_direction(self, grad: np.ndarray) -> np.ndarray:
        eps = np.sqrt(np.finfo(np.float64).resolution)
        U, s, V = svd(self._hessian)
        scale = 1 / np.maximum(s, eps)
        return U @ (V @ (-grad) * scale)


class HessianFree(Optimizer):
    def _call_to_oracle(self, w: np.ndarray) -> Tuple[float, np.ndarray]:
        val, grad = self._oracle.fuse_value_grad(w)
        self._cur_w = w
        return val, grad

    def _get_direction(self, grad: np.ndarray) -> np.ndarray:
        x, b = -grad, -grad
        r_cur = self._oracle.hessian_vec_product(self._cur_w, x) - b
        p = -r_cur
        for _ in range(2 * grad.shape[0]):
            if np.linalg.norm(r_cur) < self._tol:
                break
            r_cur_norm = r_cur @ r_cur
            A_p = self._oracle.hessian_vec_product(self._cur_w, p)
            alpha = r_cur_norm / (p @ A_p)
            x += alpha * p
            r_next = r_cur + alpha * A_p
            r_next_norm = r_next @ r_next
            beta = r_next_norm / r_cur_norm
            p = -r_next + beta * p

            r_cur, r_cur_norm = r_next, r_next_norm

        return x


class LBFGS(Optimizer):
    def __init__(
        self,
        oracle: Oracle,
        line_search: LineSearch,
        start_point: np.ndarray,
        history_size: int,
        tol: float,
        max_iter: int,
    ):
        super().__init__(oracle, line_search, start_point, tol, max_iter)
        self._history: Deque[Tuple[np.ndarray, np.ndarray]] = deque(maxlen=history_size)
        self._last_w: Optional[np.ndarray] = None
        self._last_grad: Optional[np.ndarray] = None

    def _call_to_oracle(self, w: np.ndarray) -> Tuple[float, np.ndarray]:
        value, grad = self._oracle.fuse_value_grad(w)

        if self._last_w is not None:
            s = w - self._last_w
            y = grad - self._last_grad
            self._history.appendleft((s, y))

        self._last_w = w
        self._last_grad = grad

        return value, grad

    def _get_direction(self, grad: np.ndarray) -> np.ndarray:
        d = -grad
        mus = []
        # first is the newest
        for s, y in self._history:
            mu = (s @ d) / (s @ y)
            d -= mu * y
            mus.append(mu)
        if self._history:
            s_newest, y_newest = self._history[0]
            d *= (s_newest @ y_newest) / (y_newest @ y_newest)
        for mu, (s, y) in zip(reversed(mus), reversed(self._history)):
            beta = (y @ d) / (s @ y)
            d += (mu - beta) * s
        return d
