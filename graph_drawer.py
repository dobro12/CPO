import matplotlib.pyplot as plt
from collections import deque
import multiprocessing as mp
import numpy as np
import time

class ProcessPlotter(object):
    def __init__(self, freq, title, label_instruct):
        self.interval = freq
        self.title = title
        self.label_instruct = label_instruct
        self.color = ['r','g','b']
        max_len = int(5e3)

        self.buffers = []
        for i in range(len(label_instruct)):
            if type(label_instruct[i]) == list:
                temp_buffers = []
                for j in range(len(label_instruct[i])):
                    temp_buffers.append(deque(maxlen=max_len))
                self.buffers.append(temp_buffers)
            else:
                self.buffers.append(deque(maxlen=max_len))
        self.x = deque(maxlen=max_len)

    def terminate(self):
        plt.close('all')

    def call_back(self):
        while self.pipe.poll():
            command = self.pipe.recv()
            if command is None:
                self.terminate()
                return False

            count, data = command
            self.x.append(count)
            command = data
            for i in range(len(self.label_instruct)):
                if type(self.label_instruct[i]) == list:
                    for j in range(len(self.label_instruct[i])):
                        self.buffers[i][j].append(command[i][j])
                else:
                    self.buffers[i].append(command[i])

        for ax in self.ax_list:
            del ax.lines[-1]

        if len(self.x) == 0:
            lower, upper = 0, 0
        else:
            lower, upper = self.x[0], self.x[-1]

        cnt = 0
        for i in range(len(self.label_instruct)):
            if type(self.label_instruct[i]) == list:
                for j in range(len(self.label_instruct[i])):
                    self.ax_list[i].plot(self.x, self.buffers[i][j], self.color[cnt%len(self.color)], label=self.label_instruct[i][j])
                    cnt += 1
            else:
                self.ax_list[i].plot(self.x, self.buffers[i], self.color[cnt%len(self.color)])
                cnt += 1
            self.ax_list[i].set_xlim(lower, upper)

        self.fig.canvas.draw()
        return True

    def __call__(self, pipe):
        print('starting plotter...')

        self.pipe = pipe
        #self.fig, self.ax = plt.subplots()
        num_plots = len(self.label_instruct)
        self.fig = plt.figure(figsize=(4*num_plots,4))
        self.fig.suptitle('{}'.format(self.title), fontsize=16)

        self.ax_list = []
        cnt = 0
        for i in range(num_plots):
            ax = self.fig.add_subplot(1,num_plots,i+1)
            if type(self.label_instruct[i]) == list:
                temp_title = ""
                for temp_label in self.label_instruct[i]:
                    ax.plot([], [], self.color[cnt%len(self.color)], label=temp_label)
                    cnt += 1
                    temp_title += temp_label + " & "
                temp_title = temp_title[:-3]
                ax.set_title(temp_title)
                ax.set_xlabel('iters')
                ax.set_ylabel('')
                ax.legend()
            else:
                ax.plot([], [], self.color[cnt%len(self.color)])
                cnt += 1
                temp_title = self.label_instruct[i]
                ax.set_title(temp_title)
                ax.set_xlabel('iters')
                ax.set_ylabel('')
            plt.grid()
            self.ax_list.append(ax)

        timer = self.fig.canvas.new_timer(interval=self.interval)
        timer.add_callback(self.call_back)
        timer.start()

        print('...done')
        plt.tight_layout()
        plt.show()

class Graph:
    def __init__(self, freq, title, label_instruct):
        self.plot_pipe, plotter_pipe = mp.Pipe()
        self.plotter = ProcessPlotter(freq=freq, title=title, label_instruct=label_instruct)
        self.plot_process = mp.Process(
            target=self.plotter, args=(plotter_pipe,), daemon=True)
        self.plot_process.start()

        self.count = 0

    def update(self, data_list, finished=False):
        self.count += 1

        send = self.plot_pipe.send
        if finished:
            send(None)
        else:
            send([self.count, data_list])
