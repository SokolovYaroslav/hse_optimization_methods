from typing import Tuple, List, Optional

import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from tqdm import tqdm

from logistic_regression.line_search import LineSearch
from logistic_regression.optimizers import get_optimizer
from logistic_regression.oracle import Oracle

COLORS = [
    "#1f77b4",  # muted blue
    "#ff7f0e",  # safety orange
    "#2ca02c",  # cooked asparagus green
    "#d62728",  # brick red
    "#9467bd",  # muted purple
    "#8c564b",  # chestnut brown
    "#e377c2",  # raspberry yogurt pink
    "#7f7f7f",  # middle gray
    "#bcbd22",  # curry yellow-green
    "#17becf",  # blue-teal
]

TITLES = {
    "times": "spent time",
    "calls": "oracle calls",
    "iters": "method's iterations",
    "loss_diff": r"$\log_{10}(|F(w_k) - F(w_*)|)$",
    "grad_norm": r"$\log_{10}(\frac{|\nabla F(w_k)|^2}{|\nabla F(w_0)|^2})$",
    "loss": r"$f(w_k)$",
    "lasso_stop": r"$\log_{10}(||\frac{w_k - prox_{\alpha}(w_k - \alpha * \nabla f(w_k))}{\alpha}||^2)$",
}

TITLES_REQUIRED_SMOOTHING = {"lasso_stop", "grad_norm"}

X_STATS = {"times", "calls", "iters"}


# https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
def _moving_average(a: np.ndarray, n: int = 3) -> np.ndarray:
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


class Experiment:
    @staticmethod
    def make_experiment(
        oracle: Oracle,
        optimizer: str,
        start_point: np.ndarray,
        tol: float,
        max_iter: int,
        line_search: Optional[str] = None,
        line_search_params: Optional[dict] = None,
        history_size: Optional[int] = None,
        regularization_coeff: Optional[float] = None,
    ) -> Tuple[np.ndarray, dict]:
        if line_search is not None:
            if line_search_params is None:
                line_search_params = {}
            line_search = LineSearch.get_line_search(
                line_search, oracle, **line_search_params
            )
        optimizer = get_optimizer(
            optimizer,
            oracle,
            start_point,
            tol,
            max_iter,
            line_search,
            history_size,
            regularization_coeff,
        )
        w_opt = optimizer()
        return w_opt, optimizer.stats

    @staticmethod
    def draw_figs(
        names: List[str],
        stats: List[dict],
        axes: Tuple[str, ...] = ("times", "calls", "iters"),
        enable_smoothing: bool = False,
        dataset_name: str = None,
    ):
        y_stats = set.intersection(*(set(stat.keys()) for stat in stats))
        y_stats -= X_STATS

        for ax in axes:
            fig = make_subplots(
                rows=1,
                cols=len(y_stats),
                horizontal_spacing=0.05,
                subplot_titles=[TITLES[y_stat] for y_stat in y_stats],
            )
            title = f"Error dependency by {TITLES[ax]}"
            if dataset_name:
                title += f" on dataset {dataset_name}"
            fig.update_layout(title=title, legend={"orientation": "h"})
            for i, y_stat in enumerate(y_stats):
                for name, stat, color in zip(names, stats, COLORS):
                    smooth_n = 10
                    if (
                        enable_smoothing
                        and y_stat in TITLES_REQUIRED_SMOOTHING
                        and len(stat[y_stat]) > smooth_n * 2
                    ):
                        y = _moving_average(np.array(stat[y_stat]), n=smooth_n)
                        x = stat[ax][smooth_n - 1 :]
                    else:
                        x = stat[ax]
                        y = stat[y_stat]
                    scatter = go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        name=name,
                        line_color=color,
                        showlegend=i == 0,
                        legendgroup=name,
                    )
                    fig.add_trace(scatter, row=0 + 1, col=i + 1)
            fig.show()

    @staticmethod
    def draw_lasso_figs(
        oracle: Oracle,
        start_point: np.ndarray,
        tol: float,
        max_iter: int,
        regularization_coeffs: List[float],
        enable_smoothing: bool = True,
    ):
        times = []
        iters = []
        non_zero_params = []
        stats = []
        for regularization_coeff in tqdm(regularization_coeffs):
            optimizer = get_optimizer(
                "lasso",
                oracle,
                start_point,
                tol,
                max_iter,
                regularization_coeff=regularization_coeff,
            )
            w_opt = optimizer()
            stat = optimizer.stats
            stats.append(stat)

            times.append(stat["times"][-1])
            iters.append(stat["iters"][-1])
            non_zero_params.append(w_opt.shape[0] - (w_opt == 0.0).sum())

        # time plot
        fig = px.line(
            x=regularization_coeffs,
            y=times,
            log_x=True,
            title="Spent time depending on lambda",
        )
        fig.update_layout(
            yaxis={"title": "Spent time"},
            xaxis={"title": r"$\lambda$", "exponentformat": "e"},
        )
        fig.show()

        # iters plot
        fig = px.line(
            x=regularization_coeffs,
            y=iters,
            log_x=True,
            title="Num iters depending on lambda",
        )
        fig.update_layout(
            yaxis={"title": "Num iters"},
            xaxis={"title": r"$\lambda$", "exponentformat": "e"},
        )
        fig.show()

        # non_zero_params plot
        fig = px.line(
            x=regularization_coeffs,
            y=non_zero_params,
            log_x=True,
            title="Num zero params depending on lambda",
        )
        fig.update_layout(
            yaxis={"title": "Num of nonzero params"},
            xaxis={"title": r"$\lambda$", "exponentformat": "e"},
        )
        fig.show()

        names = [
            f"$\lambda = {regularization_coeff:.2E}$"
            for regularization_coeff in regularization_coeffs
        ]
        Experiment.draw_figs(
            names,
            stats,
            axes=("iters",),
            enable_smoothing=enable_smoothing,
            dataset_name=oracle.name,
        )
