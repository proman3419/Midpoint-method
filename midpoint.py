import numpy as np
import matplotlib.pyplot as plt
from math import exp
import argparse


# LOGIC ########################################################################
def f_deriv(A, B, t, x):
    return A*x + B*exp(-t)


def calculate_C(A, B, a, xa):
    return (xa * (A+1) + B*exp(-a)) / (exp(A*a) * (A+1))


def f(A, B, C, t, xa):
    return C*exp(A*t) - ((B*exp(-t)) / (A+1))


def calculate_xs_act(A, B, C, ts, xa):
    xs_act = []

    for t in ts:
        xs_act.append(f(A, B, C, t, xa))

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
    C = calculate_C(A, B, a, xa)

    while True:
        ts = np.linspace(a, b, n)
        xs_act = calculate_xs_act(A, B, C, ts, xa)
        xs = midpoint(A, B, xa, ts)
        prec = calculate_precision(xs, xs_act)

        plt.plot(ts, xs, label='calculated')
        plt.plot(ts, xs_act, label='actual')
        plt.legend()
        plt.show()
        
        if prec <= E:
            return (xs, xs_act)

        n *= 2
# LOGIC ########################################################################


# INPUT ########################################################################
def validate_positive_int(arg_name, arg_val):
    if arg_val <= 0:
        print(f'{arg_name} is expected to be a positive int=')
        print(f'{arg_name} value: {arg_val}')
        exit()


def validate_order(arg1_name, arg1_val, arg2_name, arg2_val):
    if arg1_val >= arg2_val:
        print(f'{arg1_name} is expected to be smaller than {arg2_name}')
        print(f'{arg1_name} value: {arg1_val}, {arg2_name} value: {arg2_val}')
        exit()


def get_args():
    arg_parser = argparse.ArgumentParser(description='''Midpoint method for equations of the form x'(t) = A*x + B*exp(-t)''', formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument(
        '-A', type=float, default=-3, help='A coefficient')
    arg_parser.add_argument(
        '-B', type=float, default=2, help='B coefficient')
    arg_parser.add_argument(
        '-a', type=float, default=0, help='Start of the range to be inspected')
    arg_parser.add_argument(
        '-b', type=float, default=10, help='End of the range to be inspected')
    arg_parser.add_argument(
        '-n', type=int, default=16, help='Amount of steps')
    arg_parser.add_argument(
        '-xa', type=float, default=2, help='f(ta), initial value')
    arg_parser.add_argument(
        '-E', type=float, default=0.001, help='Precision of solution')
    
    args = arg_parser.parse_args()
    validate_positive_int('n', args.n)
    validate_order('a', args.a, 'b', args.b)

    return args
# INPUT ########################################################################


# OUTPUT #######################################################################

# OUTPUT #######################################################################


if __name__ == '__main__':
    args = get_args()
    midpoint_with_precision(args.A, args.B, args.a, args.b, args.n, args.xa, args.E)
