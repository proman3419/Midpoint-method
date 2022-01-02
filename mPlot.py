import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.widgets import Button


class MPlot:
    current_id = -1
    starting_indexes = dict()
    steps_by_id = dict()
    prev_button = None
    next_button = None

    def __init__(self, output_file_path, a, b):
        self.output_file_path = output_file_path
        self.a = a
        self.b = b
        self.starting_indexes[0] = 0
        plt.ion()
        self.fig = plt.figure()
        plt.subplots_adjust(bottom=0.2)

    def update_displayer(self, ts, xs_list, labels, steps, skip_adding=False):
        if not skip_adding:
            self.current_id += 1

        self.fig.clf()
        graph = self.fig.add_subplot(111)

        for i in range(len(labels)):
            graph.plot(ts, xs_list[i], label=labels[i])

        plt.title(f'Midpoint iteration number {self.current_id + 1} [{steps} steps]')
        plt.legend()

        if not skip_adding:
            self.steps_by_id[self.current_id] = steps
            self.starting_indexes[self.current_id + 1] = self.starting_indexes.get(self.current_id) + steps

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def next(self, event):
        self.current_id = (self.current_id + 1) % len(self.steps_by_id)
        self.update_plot_after_button_click()

    def prev(self, event):
        self.current_id = (self.current_id - 1) % len(self.steps_by_id)
        self.update_plot_after_button_click()

    def update_plot_after_button_click(self):
        steps = self.steps_by_id.get(self.current_id)
        actual, calculated = self.get_data()
        ts = np.linspace(self.a, self.b, steps)
        self.update_displayer(ts, [actual, calculated], ['actual', 'calculated'], steps, skip_adding=True)
        self.add_buttons()

    def get_data(self):
        starting_row, border_row = self.get_starting_and_border_rows(self.current_id)
        df = pd.read_csv(self.output_file_path, delimiter='|', usecols=['actual value', 'calculated value'])
        actual = df['actual value'].values.tolist()[starting_row: border_row]
        calculated = df['calculated value'].values.tolist()[starting_row: border_row]
        return actual, calculated

    def get_starting_and_border_rows(self, interation_index):
        starting_row = self.starting_indexes.get(interation_index)
        border_row = starting_row + self.steps_by_id.get(interation_index)
        return starting_row, border_row

    def add_buttons(self):
        ax_prev = plt.axes([0.7, 0.05, 0.1, 0.075])
        ax_next = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.next_button = Button(ax_next, 'Next')
        self.next_button.on_clicked(self.next)
        self.prev_button = Button(ax_prev, 'Previous')
        self.prev_button.on_clicked(self.prev)
        plt.show()

    def end_interation_and_add_buttons(self):
        plt.ioff()
        self.starting_indexes.pop(self.current_id + 1)
        self.add_buttons()
