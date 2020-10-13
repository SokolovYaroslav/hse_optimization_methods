import decimal
from math import isclose, sqrt, copysign
from typing import Tuple, Callable, Optional, List

from oracul import funcs


def optimize(f, a: float, b: float, eps: float = 1e-8):
    return brent_d(f, a, b, eps)[0]


def brent_d(func: Callable[[float], Tuple[float, float]], a: float, b: float, eps: float = 1e-8):
    max_iter = 50
    history = []
    zeps = eps * 1e-1

    x = w = v = (a + b) / 2  # x is a current minimum, w is a second minimum, v is a previous w
    f, df = func(x)
    fx = fw = fv = f
    dx = dw = dv = df
    length, diff = b - a, b - a
    for _ in range(max_iter):
        history.append(x)
        x_mid = (a + b) / 2
        tol1 = eps * abs(x) + zeps
        tol2 = 2 * tol1
        if abs(x - x_mid) <= (tol2 - (b - a) / 2):
            return x, history

        if abs(length) > tol1:
            d1, d2 = 2.0 * (b - a), 2.0 * (b - a)

            # Fitting parabolas
            ds = []
            if dw != dx:
                d1 = (w - x) * dx / (dx - dw)
                # d3 = cubic_minima(w, x, fw, fx, dw, dx)
                # print(f"par: {d1}, cubic: {d3}")
            if dv != dx:
                d2 = (v - x) * dx / (dx - dv)
                # d4 = cubic_minima(v, x, fv, fx, dv, dx)
                # print(f"par: {d2}, cubic: {d4}")

            u1, u2 = x + d1, x + d2

            is_u1_fits = dx * d1 <= 0.0 < (a - u1) * (u1 - b)
            is_u2_fits = dx * d2 <= 0.0 < (a - u2) * (u2 - b)
            length, length_prev = diff, length

            if is_u1_fits or is_u2_fits:
                if is_u1_fits and is_u2_fits:
                    diff = d1 if abs(d1) < abs(d2) else d2
                elif is_u1_fits:
                    diff = d1
                else:
                    diff = d2
                if abs(diff) <= abs(length_prev / 2):
                    u = x + diff
                    if u - a < tol2 or b - u < tol2:
                        diff = copysign(tol1, x_mid - x)
                else:
                    diff = (a - x) / 2 if dx > 0 else (b - x) / 2
            else:
                diff = (a - x) / 2 if dx > 0 else (b - x) / 2

        else:
            diff = (a - x) / 2 if dx > 0 else (b - x) / 2

        if abs(diff) >= tol1:
            u = x + diff
            fu, du = func(u)
        else:
            u = x + copysign(tol1, diff)
            fu, du = func(u)
            if fu > fx:
                return x, history

        if fu <= fx:
            if u >= x:
                a = x
            else:
                b = x
            v, fv, dv = w, fw, dw
            w, fw, dw = x, fx, dx
            x, fx, dx = u, fu, du
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or x == w:
                v, fv, dv = w, fw, dw
                w, fw, dw = u, fu, du
            elif fu < fv or v == x or v == w:
                v, fv, dv = u, fu, du

    return x, history


def parabola_minima_d(x_1: float, x_2: float, d_1: float, d_2: float) -> Optional[float]:
    if not isclose(x_1, x_2) and not isclose(d_1, d_2):
        d = d_1 * (x_2 - x_1) / (d_1 - d_2)
        u = x_1 + d
        if not isclose(x_1, u) and not isclose(x_2, u) and d * d_1 <= 0.0:
            return u


def cubic_minima(x1: float, x2: float, f1: float, f2: float, d1: float, d2: float) -> Optional[float]:
    h = x2 - x1
    F = f2 - f1
    G = (d2 - d1) * h
    c = G - 2 * (F - d1 * h)
    under_root = (G - 3 * c) ** 2 - 12 * c * d1 * h
    if under_root >= 0:
        gamma = -2 * d1 * h / ((G - 3 * c) + sqrt(under_root) + 1e-8)
        return h * gamma - h
    else:
        b = d2 * h - F
        gamma = (1 - F / b) / 2
        return h * gamma - h


def choose_best_u(x: float, a: float, b: float, eps: float, us: List[Optional[float]]) -> Optional[float]:
    us = [u for u in us if u is not None and a + eps < u < b - eps]
    return min(us, key=lambda u: abs(x - u)) if us else None


def main():
    eps = 1e-8
    total_score_1 = 0
    total_score_2 = 0
    fails_number = 0
    for no, func, (a, b), x_target in funcs:
        x_predicted, history = brent_d(func, a, b, eps)
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
