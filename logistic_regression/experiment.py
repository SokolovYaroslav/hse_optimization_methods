from typing import Tuple, List

import numpy as np
import plotly.graph_objs as go

from logistic_regression.line_search import LineSearch
from logistic_regression.optimizers import Optimizer
from logistic_regression.oracle import Oracle


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
        rks: Tuple[str, ...] = ("loss_diffs", "grad_norm"),
        axes: Tuple[str, ...] = ("times", "calls", "iters"),
    ):
        for rk in rks:
            for ax in axes:
                fig = go.Figure()
                for name, stat in zip(names, stats):
                    fig.add_trace(go.Scatter(x=stat[ax], y=stat[rk], mode="lines", name=name))
                    fig.update_layout(
                        yaxis={"title": f"{rk} error", "exponentformat": "e", "type": "log"},
                        xaxis={"title": ax},
                        title=f"{rk} error dependency by {ax}",
                    )
                fig.show()
