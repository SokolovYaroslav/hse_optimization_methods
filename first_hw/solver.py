import decimal
from math import isclose, copysign
from typing import Tuple, Callable, Optional

from first_hw.oracul import funcs


def brent_prime(func: Callable[[float], Tuple[float, float]], a: float, b: float, eps: float = 1e-8):
    zeps = eps * 1e-3
    max_iter = 500

    x = w = v = (a + b) / 2  # x is a current minimum, w is a second minimum, v is a previous w
    f, d_f = func(x)
    f_x = f_w = f_v = f
    d_x = d_w = d_v = d_f
    length_cur = length_past = b - a

    for _ in range(max_iter):
        length_past_past, length_past = length_past, length_cur
        # Trying to perform parabolic step
        u_1 = parabolic_secant(a, b, x, w, d_x, d_w, eps, length_past_past)
        u_2 = parabolic_secant(a, b, x, v, d_x, d_v, eps, length_past_past)
        if u_1 is not None and u_2 is not None:
            u = u_1 if abs(u_1 - x) < abs(u_2 - x) else u_2
        elif u_1 is not None or u_2 is not None:
            u = u_1 if u_1 is not None else u_2
        else:
            # parabolic is failed, performing bisect
            if d_x > 0:
                u = (a + x) / 2
            else:
                u = (b + x) / 2

        # if abs(u - argmin) < eps:
        #     u = argmin + copysign(eps, u - argmin)  # update by at least epsilon
        if abs(u - x) < eps:
            x = u
            break

        # update the state
        length_cur = abs(u - x)
        f_u, d_u = func(u)
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


def parabolic_secant(a: float, b: float, x_1: float, x_2: float, f_x_1: float, f_x_2: float, eps: float, length: float) -> Optional[float]:
    if x_1 > x_2:
        x_1, x_2 = x_2, x_1
        f_x_1, f_x_2 = f_x_2, f_x_1
    if not isclose(x_1, x_2) and not isclose(f_x_1, f_x_2):
        u = -f_x_1 * (x_2 - x_1) / (f_x_2 - f_x_1) + x_1
        if a + eps < u < b - eps and abs(u - x_1) < length / 2 and not isclose(x_1, u) and not isclose(x_2, u):
            return u


def main():
    eps = 1e-8
    total_score = 0
    fails_number = 0
    for no, func, (a, b), x_target in funcs:
        x_predicted = brent_prime(func, a, b, eps)
        if x_target is not None:
            if abs(x_target - x_predicted) < 10 ** max(decimal.Decimal(str(x_target)).as_tuple().exponent, -8):
                print(f"#{no} -- Successfully found x with {func.calls} oracul calls")
            else:
                print(f"#{no} -- FAILED: x_target: {x_target}    |     x_predicted: {x_predicted}")
                fails_number += 1
            total_score += func.calls
        else:
            print(f"#{no} -- Found x: {x_predicted} with {func.calls} oracul calls, y is {func(x_predicted)[0]}")
    print(f"TOTAL SCORE IS {total_score} WITH {fails_number} FAILS")


if __name__ == "__main__":
    main()
