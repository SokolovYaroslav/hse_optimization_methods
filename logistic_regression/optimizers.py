import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, Tuple, Optional, Deque

from scipy.linalg import cho_factor, LinAlgError, cho_solve, svd
from sklearn.linear_model import LogisticRegression

from logistic_regression.line_search import LineSearch
from logistic_regression.oracle import Oracle
import numpy as np


def get_optimizer(
    name: str,
    oracle: Oracle,
    start_point: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 10000,
    line_search: Optional[LineSearch] = None,
    history_size: Optional[int] = None,
) -> "AbstractOptimizer":
    if name != "l-bfgs":
        assert (
            history_size is None
        ), "history_size is an argument only for l-bfgs optimizer"
    if name != "lasso":
        assert line_search is not None, "you must provide line_search method"

    if name == "GD":
        return GradientDescent(oracle, line_search, start_point, tol, max_iter)

    elif name == "newton-chol":
        line_search.turn_off_brackets()
        return NewtonCholecky(oracle, line_search, start_point, tol, max_iter)

    elif name == "newton-svd":
        line_search.turn_off_brackets()
        return NewtonSVD(oracle, line_search, start_point, tol, max_iter)

    elif name == "hessian-free":
        line_search.turn_off_brackets()
        return HessianFree(oracle, line_search, start_point, tol, max_iter)

    elif name == "l-bfgs":
        assert (
            history_size is not None
        ), f"You must provide a history size, when using l-bfgs"
        line_search.turn_off_brackets()
        return LBFGS(oracle, line_search, start_point, history_size, tol, max_iter)

    else:
        raise ValueError(f"Unknown optimizer {name}")


class AbstractOptimizer(ABC):
    def __init__(
        self,
        oracle: Oracle,
        start_point: np.ndarray,
        tol: float = 1e-8,
        max_iter: int = 10000,
    ):
        self._oracle = oracle
        self._start_point = start_point.copy()
        self._tol = tol
        self._max_iter = max_iter

        self._stats = {
            "times": [],
            "calls": [],
            "iters": [],
            "loss_diffs": [],
        }
        self._timer = None
        self._num_iterations = 0
        self._opt_loss = self._get_opt_loss(oracle, tol, max_iter)

    @abstractmethod
    def _get_opt_loss(self, oracle, tol, max_iter) -> float:
        raise NotImplementedError

    def __call__(self) -> np.ndarray:
        self._start_timer()
        return self._optimize()

    @abstractmethod
    def _optimize(self) -> np.ndarray:
        """Must call self._log on each step"""
        raise NotImplementedError

    @property
    def stats(self) -> Dict[str, list]:
        return self._stats

    def _start_timer(self) -> None:
        self._oracle.reset_calls()
        self._timer = time.time()

    def _log(self, cur_loss: float, **kwargs) -> None:
        spent_time = time.time() - self._timer
        self._num_iterations += 1
        loss_diff = abs(cur_loss - self._opt_loss)

        self._stats["times"].append(spent_time)
        self._stats["oracle_calls"].append(self._oracle.num_calls)
        self._stats["iters"].append(self._num_iterations)
        self._stats["loss_diffs"].append(np.log10(loss_diff))

        for name, val in kwargs:
            if name not in self._stats:
                self._stats[name] = []
            self._stats[name].append(val)


class CasualOptimizer(AbstractOptimizer):
    def __init__(
        self,
        oracle: Oracle,
        line_search: LineSearch,
        start_point: np.ndarray,
        tol: float = 1e-8,
        max_iter: int = 10000,
    ):
        super().__init__(oracle, start_point, tol, max_iter)
        self._line_search = line_search

    def _optimize(self) -> np.ndarray:
        w = self._start_point
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
            self._log(loss_diff, grad_norm=grad_norm)
            if grad_norm <= self._tol:
                break
        return w

    @abstractmethod
    def _call_to_oracle(self, w: np.ndarray) -> Tuple[float, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def _get_direction(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplemented

    def _get_opt_loss(self, oracle: Oracle, tol: float, max_iter: int) -> float:
        cls = LogisticRegression(
            penalty="none",
            tol=tol,
            fit_intercept=False,
            solver="newton-cg",
            max_iter=max_iter,
        )
        cls.fit(*oracle.data)
        return cls.coef_[0]


class GradientDescent(CasualOptimizer):
    def _call_to_oracle(self, w: np.ndarray) -> Tuple[float, np.ndarray]:
        return self._oracle.fuse_value_grad(w)

    def _get_direction(self, grad: np.ndarray) -> np.ndarray:
        return -grad


class NewtonCholecky(CasualOptimizer):
    def __init__(
        self,
        oracle: Oracle,
        line_search: LineSearch,
        start_point: np.ndarray,
        tol: float = 1e-8,
        max_iter: int = 10000,
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


class NewtonSVD(CasualOptimizer):
    def _call_to_oracle(self, w: np.ndarray) -> Tuple[float, np.ndarray]:
        val, grad, hessian = self._oracle.fuse_value_grad_hessian(w)
        self._hessian = hessian
        return val, grad

    def _get_direction(self, grad: np.ndarray) -> np.ndarray:
        eps = np.sqrt(np.finfo(np.float64).resolution)
        U, s, V = svd(self._hessian)
        scale = 1 / np.maximum(s, eps)
        return U @ (V @ (-grad) * scale)


class HessianFree(CasualOptimizer):
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


class LBFGS(CasualOptimizer):
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


class LassoOptimizer(AbstractOptimizer):
    def __init__(
        self,
        oracle: Oracle,
        start_point: np.ndarray,
        lambda_: float,
        tol: float = 1e-8,
        max_iter: int = 10000,
    ):
        assert max_iter > 0
        super().__init__(oracle, start_point, tol, max_iter)
        self._lambda = lambda_

    def _optimize(self) -> np.ndarray:
        w = self._start_point
        lipschitz = 1.0

        loss, grad = self._oracle.fuse_value_grad(w)
        for _ in range(self._max_iter):
            for _ in range(self._max_iter):
                alpha = 1 / lipschitz
                new_w = self._proximal_step(w - alpha * grad, alpha)
                new_loss = self._oracle.value(new_w)
                ws_diff = new_w - w
                ws_diff_norm2 = ws_diff @ ws_diff
                if new_loss <= loss + (grad @ ws_diff) + lipschitz / 2 * ws_diff_norm2:
                    break
                lipschitz *= 2

            w, loss = new_w, new_loss

            self._log(loss)
            if ws_diff_norm2 / (alpha ** 2) <= self._tol:
                break

            lipschitz /= 2

        return w

    def _proximal_step(self, x: np.ndarray, alpha: float) -> np.ndarray:
        alpha = self._lambda * alpha
        return np.sign(x) * np.maximum(np.abs(x) - alpha, 0)

    def _get_opt_loss(self, oracle: Oracle, tol: float, max_iter: int) -> float:
        cls = LogisticRegression(
            penalty="l1",
            C=1 / self._lambda,
            tol=tol,
            fit_intercept=False,
            solver="liblinear",
            max_iter=max_iter,
        )
        cls.fit(*oracle.data)
        return cls.coef_[0]
