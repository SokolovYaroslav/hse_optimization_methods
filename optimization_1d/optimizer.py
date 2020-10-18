import decimal
from math import copysign
from typing import Tuple, Callable

from oracul import funcs


def optimize(f, a: float, b: float, eps: float = 1e-8):
    return brent_d(f, a, b, eps)[0]


def brent_d(
    func: Callable[[float], Tuple[float, float]], a: float, b: float, eps: float = 1e-8
):
    max_iter = 500
    history = []
    zeps = eps * 1e-1

    # x is a current minimum, w is a second minimum, v is a previous w
    x = w = v = (a + b) / 2
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
            if dw != dx:
                d1 = (w - x) * dx / (dx - dw)
            if dv != dx:
                d2 = (v - x) * dx / (dx - dv)

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
            u = x + copysign(tol1, diff)  # adjusting at least by tol1
            fu, du = func(u)
            if fu > fx:
                return x, history

        # Updating the state
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


def main():
    eps = 1e-8
    total_score_1 = 0
    total_score_2 = 0
    fails_number = 0
    for no, func, (a, b), x_target in funcs:
        x_predicted, history = brent_d(func, a, b, eps)
        if x_target is not None:
            if abs(x_target - x_predicted) < 10 ** max(
                decimal.Decimal(str(x_target)).as_tuple().exponent, -8
            ):
                print(f"#{no} -- Successfully found x with {func.calls} oracul calls")
            else:
                print(
                    f"#{no} -- FAILED: x_target: {x_target}    |     x_predicted: {x_predicted}"
                )
                fails_number += 1
            total_score_1 += func.calls
        else:
            print(
                f"#{no} -- Found x: {x_predicted} with {func.calls} oracul calls, y is {func(x_predicted)[0]}"
            )
            total_score_2 += func.calls
    print(
        f"TOTAL SCORE FOR UNI IS {total_score_1} FOR NON-UNI IS {total_score_2} WITH {fails_number} FAILS"
    )


if __name__ == "__main__":
    main()
