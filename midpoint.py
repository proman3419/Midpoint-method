import numpy as np
import matplotlib.pyplot as plt
from math import exp


def f_deriv(A, B, x, t):
    return A*x + B*exp(-t)


def f(A, B, t):
    # return (exp(-A*t) * ((B * (exp((A-1)*t)) - 1) + 2*A - 2)) / (A-1)
    return (exp(2*t) + 1) * exp(-3*t)


def calculate_xs_act(A, B, ts):
    xs_act = []

    for t in ts:
        xs_act.append(f(A, B, t))

    return xs_act


def midpoint(A, B, xa, ts):
    xk = xa
    xs = [xa]

    for i in range(len(ts)-1):
        h = ts[i+1] - ts[i]
        xk1 = xk + h/2 * f_deriv(A, B, ts[i], xk)
        xk2 = xk + h * f_deriv(A, B, ts[i] + h/2, xk1)
        xk = xk2
        xs.append(xk2)

    return xs


def calculate_precision(xs, xs_act):
    n = len(xs)
    return sum(abs(xs[i] - xs_act[i]) for i in range(n)) / n


def display_graph(ts, xs, series_label):
    plt.plot(ts, xs, label=series_label)


def midpoint_with_precision(A, B, a, b, n, xa, E):
    i = 10

    while True:
        ts = np.linspace(a, b, n)
        print('ts', ts)
        xs_act = calculate_xs_act(A, B, ts)
        print('xs_act', xs_act)
        xs = midpoint(A, B, xa, ts)
        print('xs', xs)
        prec = calculate_precision(xs, xs_act)
        print('prec', prec)
        print(n, prec)
        print()

        if prec <= E:
            return (xs, xs_act)

        n *= 2

        display_graph(ts, xs, 'calculated')
        display_graph(ts, xs_act, 'actual')
        plt.legend()
        plt.show()

        i -= 1
        if i <= 0:
            break


if __name__ == '__main__':
    print(midpoint_with_precision(-3, 2, 0, 10, 10, 2, 0.1))
