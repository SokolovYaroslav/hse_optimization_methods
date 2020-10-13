import decimal
from math import isclose, sqrt, copysign
from typing import Tuple, Callable, Optional, List

from first_hw.oracul import funcs


def brent_d(func: Callable[[float], Tuple[float, float]], a: float, b: float, eps: float = 1e-8):
    max_iter = 50
    zeps = eps * 1e-3

    x = w = v = (a + b) / 2  # x is a current minimum, w is a second minimum, v is a previous w
    f, d_f = func(x)
    f_x = f_w = f_v = f
    d_x = d_w = d_v = d_f
    length_cur = length_past = b - a
    for _ in range(max_iter):
        # if isclose(d_x, 0.0):
        #     return x
        x_med = (a + b) / 2
        tol1 = eps * abs(x) + zeps
        tol2 = 2 * tol1
        if abs(x - x_med) <= (tol2 - (b - a) / 2):
            return x

        length_past_past, length_past = length_past, length_cur

        # Trying to preform a parabola and cubic minimization
        us = [
            parabola_minima_d(x, w, d_x, d_w),
            parabola_minima_d(x, v, d_x, d_v),
            cubic_minima(x, w, f_x, f_w, d_x, d_w),
            cubic_minima(x, v, f_x, f_v, d_x, d_v)
        ]
        u = choose_optimum(x, a, b, eps, us)

        if u is None:
            # Super-linear methods fail, perform bisect
            u = (a + x) / 2 if d_x > 0 else (b + x) / 2

        if abs(u - x) < eps:
            return x

        if abs(u - x) < tol1:
            u = x + copysign(tol1, u - x)
            f_u, d_u = func(u)
            if f_u > f_x:
                return x
        else:
            f_u, d_u = func(u)

        # update the state
        length_cur = abs(u - x)
        if f_u < f_x:
            if u >= x:
                a = x
            else:
                b = x
            x, w, v = u, x, w
            f_x, f_w, f_v = f_u, f_x, f_w
            d_x, d_w, d_v = d_u, d_x, d_w
        else:
            if u >= x:
                b = u
            else:
                a = u
            if f_u <= f_w or isclose(x, w):
                w, v = u, w
                f_w, f_v = f_u, f_w
                d_w, d_v = d_u, d_w
            elif f_u <= f_v or isclose(x, v) or isclose(w, v):
                v = u
                f_v = f_u
                d_v = f_u

    return x


def parabola_minima(x_1: float, x_2: float, x_3: float, f_1: float, f_2: float, f_3: float) -> Optional[float]:
    if f_1 < f_2:
        x_1, x_2 = x_2, x_1
        f_1, f_2 = f_2, f_1
    if f_3 < f_2:
        x_3, x_2 = x_2, x_3
        f_3, f_2 = f_2, f_3
    if not (isclose(x_1, x_2) or isclose(x_2, x_3) or isclose(x_1, x_3) or
            isclose(f_1, f_2) or isclose(f_2, f_3) or isclose(f_1, f_3)):
        u = (x_2 -
             ((x_2 - x_1) ** 2 * (f_2 - f_3) - (x_2 - x_3) ** 2 * (f_2 - f_1)) /
             (2 * (x_2 - x_1) * (f_2 - f_3) - (x_2 - x_3) * (f_2 - f_1))
             )
        if not (isclose(x_1, u) or isclose(x_2, u) or isclose(x_3, u)):
            return u


def parabola_minima_d(x_1: float, x_2: float, d_1: float, d_2: float) -> Optional[float]:
    if x_1 > x_2:
        x_1, x_2 = x_2, x_1
        d_1, d_2 = d_2, d_1
    if not isclose(x_1, x_2) and not isclose(d_1, d_2):
        u = -d_1 * (x_2 - x_1) / (d_2 - d_1) + x_1
        if not isclose(x_1, u) and not isclose(x_2, u):
            return u


def cubic_minima(x_1: float, x_2: float, f_1: float, f_2: float, d_1: float, d_2: float) -> Optional[float]:
    if x_1 > x_2:
        x_1, x_2 = x_2, x_1
        f_1, f_2 = f_2, f_1
        d_1, d_2 = d_2, d_1
    if not isclose(x_1, x_2) and not isclose(f_1, f_2) and not isclose(d_1, d_2):
        h = x_2 - x_1
        F = f_2 - f_1
        G = (d_2 - d_1) * h
        c = G - 2 * (F - d_1 * h)
        under_root = (G - 3 * c) ** 2 - 12 * c * d_1 * h
        if under_root >= 0:
            u = -2 * d_1 * h / ((G - 3 * c) + sqrt(under_root) + 1e-8)
            if not isclose(x_1, u) and not isclose(x_2, u):
                return u


def choose_optimum(x: float, a: float, b: float, eps: float, us: List[Optional[float]]) -> Optional[float]:
    us = [u for u in us if u is not None and a + eps < u < b - eps]
    return min(us, key=lambda u: abs(x - u)) if us else None


def main():
    eps = 1e-8
    total_score_1 = 0
    total_score_2 = 0
    fails_number = 0
    for no, func, (a, b), x_target in funcs:
        x_predicted = brent_d(func, a, b, eps)
        if x_target is not None:
            if abs(x_target - x_predicted) < 10 ** max(decimal.Decimal(str(x_target)).as_tuple().exponent, -8):
                print(f"#{no} -- Successfully found x with {func.calls} oracul calls")
            else:
                print(f"#{no} -- FAILED: x_target: {x_target}    |     x_predicted: {x_predicted}")
                fails_number += 1
            total_score_1 += func.calls
        else:
            print(f"#{no} -- Found x: {x_predicted} with {func.calls} oracul calls, y is {func(x_predicted)[0]}")
            total_score_2 += func.calls
    print(f"TOTAL SCORE FOR UNI IS {total_score_1} FOR NON-UNI IS {total_score_2} WITH {fails_number} FAILS")


if __name__ == "__main__":
    main()
