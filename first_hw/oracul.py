from math import sin, exp, pi, log, cos
from typing import Callable, List, Tuple, Optional


def _calc_prime(func: Callable[[float], float], x: float, eps: float = 1e-8) -> float:
    return (func(x + eps) - func(x - eps)) / (2 * eps)


def _func_wrapper(func: Callable[[float], float]) -> Callable[[float], Tuple[float, float]]:
    def inner(x: float) -> Tuple[float, float]:
        inner.calls += 1
        return func(x), _calc_prime(func, x)
    inner.calls = 0
    return inner


funcs: List[Tuple[int, Callable[[float], Tuple[float, float]], Tuple[float, float], Optional[float]]] = [
    (0, _func_wrapper(lambda x: x * x), (-1000, 239), 0.0),
    # 02
    (2, _func_wrapper(lambda x: sin(x) + sin(10*x/3)), (2.7, 7.5), None),
    (2, _func_wrapper(lambda x: sin(x) + sin(10*x/3)), (4.5, 6.0), 5.145735),
    # 03
    (3, _func_wrapper(lambda x: -sum(k*sin((k+1)*x+k) for k in range(1, 6))), (-10, 10), None),
    # 04
    (4, _func_wrapper(lambda x: -(16*x**2 -24*x + 5)*exp(-x)), (1.9, 3.9), 2.868034),
    # 05
    (5, _func_wrapper(lambda x: -(1.4 - 3*x)*sin(18*x)), (0, 1.2), None),
    (5, _func_wrapper(lambda x: -(1.4 - 3*x)*sin(18*x)), (0.8, 1.1), 0.96609),
    # 06
    (6, _func_wrapper(lambda x: -(x + sin(x)) * exp(-x**2)), (-10, 10), 0.67956),
    # 07
    (7, _func_wrapper(lambda x: sin(x) + sin(10*x/3) + log(x) -0.84*x + 3), (2.7, 7.5), None),
    (7, _func_wrapper(lambda x: sin(x) + sin(10*x/3) + log(x) -0.84*x + 3), (4.2, 6.1), 5.19978),
    # 09
    (9, _func_wrapper(lambda x: sin(x) + sin(2*x/3)), (3.1, 20.4), None),
    (9, _func_wrapper(lambda x: sin(x) + sin(2*x/3)), (13, 20.4), 17.039),
    # 10
    (10, _func_wrapper(lambda x: -x*sin(x)), (0, 10), None),
    (10, _func_wrapper(lambda x: -x*sin(x)), (5, 10), 7.9787),
    # 11
    (11, _func_wrapper(lambda x: 2*cos(x) + cos(2*x)), (-pi/2, 2*pi), None),
    (11, _func_wrapper(lambda x: 2*cos(x) + cos(2*x)), (0, 3), 2.09439),
    # 12
    (12, _func_wrapper(lambda x: sin(x)**3 + cos(x)**3), (0, 2*pi), None),
    (12, _func_wrapper(lambda x: sin(x)**3 + cos(x)**3), (1.7, 6), pi),
    # 13
    (13, _func_wrapper(lambda x: -x**(2/3) - (1 - x**2)**(1/3)), (0.001, 0.99), 1/2**0.5),
    # 14
    (14, _func_wrapper(lambda x: -exp(-x)*sin(2*pi*x)), (0, 4), None),
    (14, _func_wrapper(lambda x: -exp(-x)*sin(2*pi*x)), (0, 0.6), 0.224885),
    # 15
    (15, _func_wrapper(lambda x: (x**2-5*x+6)/(x**2 + 1)), (-5, 5), None),
    (15, _func_wrapper(lambda x: (x**2-5*x+6)/(x**2 + 1)), (-0.1, 5), 2.41422),
    # 18
    (18, _func_wrapper(lambda x: (x - 2)**2 if x<=3 else 2*log(x-2)+1), (0, 6), 2),
    # 20
    (20, _func_wrapper(lambda x: -(x-sin(x))*exp(-x**2)), (-10, 10), 1.195137),
    # 21
    (21, _func_wrapper(lambda x: x*sin(x)+x*cos(2*x)), (0, 10), None),
    (21, _func_wrapper(lambda x: x*sin(x)+x*cos(2*x)), (3, 6.5), 4.79507),
    # 22
    (22, _func_wrapper(lambda x: exp(-3*x) - sin(x)**3), (0, 20), None),
    (22, _func_wrapper(lambda x: exp(-3*x) - sin(x)**3), (12, 16), 9*pi/2),
]
