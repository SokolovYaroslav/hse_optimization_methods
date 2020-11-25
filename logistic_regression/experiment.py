from typing import Tuple, List

import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from logistic_regression.line_search import LineSearch
from logistic_regression.optimizers import Optimizer
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
}


class Experiment:
    @staticmethod
    def make_experiment(
        oracle: Oracle,
        line_search: str,
        line_search_params: dict,
        optimizer: str,
        start_point: np.ndarray,
        tol: float,
        max_iter: int,
    ) -> Tuple[np.ndarray, dict]:
        line_search = LineSearch.get_line_search(line_search, oracle, **line_search_params)
        optimizer = Optimizer.get_optimizer(optimizer, oracle, line_search, start_point, tol, max_iter)
        w_opt = optimizer()
        return w_opt, optimizer.stats

    @staticmethod
    def draw_figs(
        names: List[str],
        stats: List[dict],
        axes: Tuple[str, ...] = ("times", "calls", "iters"),
    ):
        for ax in axes:
            fig = make_subplots(
                rows=1,
                cols=2,
                horizontal_spacing=0.05,
                subplot_titles=[r"$\log(|F(w_k) - F(w_*)|)$", r"$\log(\frac{|\nabla F(w_k)|^2}{|\nabla F(w_0)|^2})$"],
            )
            fig.update_layout(title=f"Error dependency by {TITLES[ax]}", legend={"orientation": "h"})
            for i, rk in enumerate(("loss_diffs", "grad_norm")):
                for name, stat, color in zip(names, stats, COLORS):
                    scatter = go.Scatter(
                        x=stat[ax],
                        y=stat[rk],
                        mode="lines",
                        name=name,
                        line_color=color,
                        showlegend=i == 0,
                        legendgroup=name,
                    )
                    fig.add_trace(scatter, row=0 + 1, col=i + 1)
            fig.show()
