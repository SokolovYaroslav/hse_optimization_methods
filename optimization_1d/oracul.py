from math import sin, exp, pi, log, cos
from typing import Callable, List, Tuple, Optional


f0 = lambda x: (x ** 2 / 2, x)

f2 = lambda x: (sin(x) + sin(10 / 3 * x), cos(x) + 10 / 3 * cos(10 / 3 * x))

f3 = lambda x: (
    -sum(k * sin((k + 1) * x + k) for k in range(1, 6 + 1)),
    -sum(k * (k + 1) * cos((k + 1) * x + k) for k in range(1, 6 + 1)),
)

f4 = lambda x: (
    -(16 * x ** 2 - 24 * x + 5) * exp(-x),
    (16 * x ** 2 - 56 * x + 29) * exp(-x),
)

f5 = lambda x: (
    -(1.4 - 3 * x) * sin(18 * x),
    3 * (18 * (x - 0.466667) * cos(18 * x) + sin(18 * x)),
)

f6 = lambda x: (
    -(x + sin(x)) * exp(-(x ** 2)),
    exp(-(x ** 2)) * (2 * x ** 2 + 2 * x * sin(x) - cos(x) - 1),
)

f7 = lambda x: (
    sin(x) + sin(10 / 3 * x) + log(x) - 0.84 * x + 3,
    1 / x + cos(x) + 10 / 3 * cos(10 / 3 * x) - 0.84,
)

f9 = lambda x: (sin(x) + sin(2 / 3 * x), cos(x) + 2 / 3 * cos(2 / 3 * x))

f10 = lambda x: (-x * sin(x), -sin(x) - x * cos(x))

f11 = lambda x: (2 * cos(x) + cos(2 * x), -2 * (sin(x) + sin(2 * x)))

f12 = lambda x: (sin(x) ** 3 + cos(x) ** 3, 3 * sin(x) * cos(x) * (sin(x) - cos(x)))

f18 = lambda x: (
    ((x - 2) ** 2, 2 * (x - 2)) if x <= 3 else (2 * log(x - 2) + 1, 2 / (x - 2))
)


def _func_wrapper(
    func: Callable[[float], Tuple[float, float]]
) -> Callable[[float], Tuple[float, float]]:
    def inner(x: float) -> Tuple[float, float]:
        inner.calls += 1
        return func(x)

    inner.calls = 0
    return inner


funcs: List[
    Tuple[
        int,
        Callable[[float], Tuple[float, float]],
        Tuple[float, float],
        Optional[float],
    ]
] = [
    (0, _func_wrapper(f0), (-5, 5), 0.0),
    # 02
    (2, _func_wrapper(f2), (2.7, 7.5), None),
    (2, _func_wrapper(f2), (4.5, 6.0), 5.145735),
    # 03
    (3, _func_wrapper(f3), (-10, 10), None),
    # 04
    (4, _func_wrapper(f4), (1.9, 3.9), 2.868034),
    # 05
    (5, _func_wrapper(f5), (0, 1.2), None),
    (5, _func_wrapper(f5), (0.8, 1.1), 0.96609),
    # 06
    (6, _func_wrapper(f6), (-10, 10), 0.67957),
    # 07
    (7, _func_wrapper(f7), (2.7, 7.5), None),
    (7, _func_wrapper(f7), (4.2, 6.1), 5.19978),
    # 09
    (9, _func_wrapper(f9), (3.1, 20.4), None),
    (9, _func_wrapper(f9), (13, 20.4), 17.039),
    # 10
    (10, _func_wrapper(f10), (0, 10), None),
    (10, _func_wrapper(f10), (5, 10), 7.9787),
    # 11
    (11, _func_wrapper(f11), (-pi / 2, 2 * pi), None),
    (11, _func_wrapper(f11), (0, 3), 2.09439),
    # 12
    (12, _func_wrapper(f12), (0, 2 * pi), None),
    (12, _func_wrapper(f12), (1.7, 6), pi),
    # 18
    (18, _func_wrapper(f18), (0, 6), 2),
]
