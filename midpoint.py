import numpy as np
from math import exp
from mPlot import MPlot
import argparse
import time
import csv
import os.path


# INPUT ########################################################################
def validate_int_greater_than(lower_bound, arg_name, arg_val):
    if arg_val <= lower_bound:
        print(f'{arg_name} is expected to be an int greater than {lower_bound}')
        print(f'{arg_name} value: {arg_val}')
        exit()


def validate_order(arg1_name, arg1_val, arg2_name, arg2_val):
    if arg1_val >= arg2_val:
        print(f'{arg1_name} is expected to be smaller than {arg2_name}')
        print(f'{arg1_name} value: {arg1_val}, {arg2_name} value: {arg2_val}')
        exit()


def get_args():
    arg_parser = argparse.ArgumentParser(
        description='''Midpoint method for equations of the form x'(t) = A*x + B*exp(-t)''',
        formatter_class=argparse.RawTextHelpFormatter)

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
        '-xa', type=float, default=2, help='f(a), initial value')
    arg_parser.add_argument(
        '-E', type=float, default=0.001, help='Precision of solution')

    args = arg_parser.parse_args()
    validate_int_greater_than(1, 'n', args.n)
    validate_order('a', args.a, 'b', args.b)

    return args


# INPUT ########################################################################


# OUTPUT #######################################################################
def get_output_data(n, xs_act, xs, prec, id):
    N = len(xs_act)
    data = [None] * N

    for i in range(N):
        data[i] = {'id': id,
                   'steps': n,
                   'precision': prec,
                   'actual value': xs_act[i],
                   'calculated value': xs[i]}

    return data


def print_output_data(data):
    for r in data:
        print(str(r))
    print()


def append_to_file(path, data):
    file_exists = os.path.isfile(path)
    with open(path, 'a+') as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys(), delimiter='|')
        if not file_exists:
            writer.writeheader()
        writer.writerows(data)


def output(output_file_path, fig, n, ts, xs_act, xs, prec, steps, id):
    output_data = get_output_data(n, xs_act, xs, prec, id)
    print_output_data(output_data)
    append_to_file(output_file_path, output_data)
    fig.update_displayer(ts, [xs_act, xs], ['actual', 'calculated'], steps)
# OUTPUT #######################################################################


# LOGIC ########################################################################
def f_deriv(A, B, t, x):
    return A * x + B * exp(-t)


def calculate_C(A, B, a, xa):
    return (xa * (A + 1) + B * exp(-a)) / (exp(A * a) * (A + 1))


def f(A, B, C, t):
    return C * exp(A * t) - ((B * exp(-t)) / (A + 1))


def calculate_xs_act(A, B, C, ts):
    N = len(ts)
    xs_act = np.zeros(N)

    for i in range(N):
        xs_act[i] = f(A, B, C, ts[i], xa)

    return xs_act


def midpoint(A, B, xa, ts):
    N = len(ts)
    xs = np.zeros(N)
    xs[0] = xa

    for i in range(N - 1):
        h = ts[i + 1] - ts[i]
        xk1 = xs[i] + h / 2 * f_deriv(A, B, ts[i], xs[i])
        xk2 = xs[i] + h * f_deriv(A, B, ts[i] + h / 2, xk1)
        xs[i + 1] = xk2

    return xs


def calculate_precision(xs, xs_act):
    return np.sum(np.abs(np.subtract(xs, xs_act, dtype=np.float))) / xs.shape[0]


def midpoint_iteration(A, B, C, a, b, n, xa):
    ts = np.linspace(a, b, n)
    xs_act = calculate_xs_act(A, B, C, ts)
    xs = midpoint(A, B, xa, ts)
    prec = calculate_precision(xs, xs_act)

    return ts, xs_act, xs, prec


def midpoint_with_precision(A, B, a, b, n, xa, E):
    C = calculate_C(A, B, a, xa)
    ts = np.linspace(a, b, n)
    xs_act = calculate_xs_act(A, B, C, ts)
    output_file_path = f'midpoint_report_{time.strftime("%Y-%m-%d_%H-%M-%S")}.csv'
    fig = MPlot(output_file_path, a, b)
    fig.update_displayer(ts, [xs_act], ['actual'], 0, True)

    id = 0
    while True:
        ts, xs_act, xs, prec = midpoint_iteration(A, B, C, a, b, n, xa)

        output(output_file_path, fig, n, ts, xs_act, xs, prec, n, id)
        if prec <= E:
            fig.end_interation_and_add_buttons()
            return xs, xs_act

        n *= 2
        id += 1
        time.sleep(1)
# LOGIC ########################################################################


if __name__ == '__main__':
    args = get_args()
    try:
        midpoint_with_precision(args.A, args.B, args.a, args.b, args.n, args.xa, args.E)
    except ZeroDivisionError:
        print('Zero division error')
    except OverflowError:
        print('Too large values')
    except:
        print('Error occurred')
