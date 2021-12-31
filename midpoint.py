import numpy as np
import matplotlib.pyplot as plt
from math import exp


def f_deriv(A, B, t, x):
    return A*x + B*exp(-t)


def f(A, B, t, xa):
    return (exp(A*t) * (-B * exp(-(A+1)*t) + xa*A + B + xa)) / (A+1)


def calculate_xs_act(A, B, ts, xa):
    xs_act = []

    for t in ts:
        xs_act.append(f(A, B, t, xa))

    return xs_act


def midpoint(A, B, xa, ts):
    N = len(ts)
    xs = [0]*N
    xs[0] = xa

    for i in range(N-1):
        h = ts[i+1] - ts[i]
        xk1 = xs[i] + h/2 * f_deriv(A, B, ts[i], xs[i])
        xk2 = xs[i] + h * f_deriv(A, B, ts[i] + h/2, xk1)
        xs[i+1] = xk2

    return xs


def calculate_precision(xs, xs_act):
    n = len(xs)
    return sum(abs(xs[i] - xs_act[i]) for i in range(n)) / n


def midpoint_with_precision(A, B, a, b, n, xa, E):
    while True:
        ts = np.linspace(a, b, n)
        xs_act = calculate_xs_act(A, B, ts, xa)
        xs = midpoint(A, B, xa, ts)
        prec = calculate_precision(xs, xs_act)

        plt.plot(ts, xs, label='calculated')
        plt.plot(ts, xs_act, label='actual')
        plt.legend()
        plt.show()
        
        if prec <= E:
            return (xs, xs_act)

        n *= 2


if __name__ == '__main__':
    print(midpoint_with_precision(-5, 4, 0, 10, 16, 2, 0.0001))
