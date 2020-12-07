from typing import Tuple, List, Optional

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


# https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
def _moving_average(a: np.ndarray, n: int = 3) -> np.ndarray:
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


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
        history_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, dict]:
        line_search = LineSearch.get_line_search(line_search, oracle, **line_search_params)
        optimizer = Optimizer.get_optimizer(optimizer, oracle, line_search, start_point, tol, max_iter, history_size)
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
        for ax in axes:
            fig = make_subplots(
                rows=1,
                cols=2,
                horizontal_spacing=0.05,
                subplot_titles=[
                    r"$\log_{10}(|F(w_k) - F(w_*)|)$",
                    r"$\log_{10}(\frac{|\nabla F(w_k)|^2}{|\nabla F(w_0)|^2})$",
                ],
            )
            title = f"Error dependency by {TITLES[ax]}"
            if dataset_name:
                title += f" on dataset {dataset_name}"
            fig.update_layout(title=title, legend={"orientation": "h"})
            for i, rk in enumerate(("loss_diffs", "grad_norm")):
                for name, stat, color in zip(names, stats, COLORS):
                    if enable_smoothing and i == 1:
                        y = _moving_average(np.array(stat[rk]), n=30)
                        x = stat[ax][29:]
                    else:
                        x = stat[ax]
                        y = stat[rk]
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
