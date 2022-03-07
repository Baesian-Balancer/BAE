from cProfile import label
import os
import matplotlib.pyplot as plt

class PlotUtils:
    def __init__(self,cp_name,dir):
        self.name = cp_name
        self.dir = dir
        self.hip_action = []
        self.knee_action = []

        self.hip_temporal_change = []
        self.knee_temporal_change = []

        if not os.path.isdir(dir):
            os.makedirs(dir)

    def add_action(self,action):
        if len(self.hip_action) > 0:
            self.hip_temporal_change.append(abs(self.hip_action[-1]-action[0]))
            self.knee_temporal_change.append(abs(self.knee_action[-1]-action[1]))
        self.hip_action.append(action[0])
        self.knee_action.append(action[1])


    def plot_action_histogram(self):
        if not os.path.isdir(self.dir):
            os.mkdir(self.dir)
        plt.figure()
        plt.hist(self.hip_action)
        plt.title("Hip action across episode")
        PATH = self.dir + self.name + "_hip_action_histogram.png"
        plt.savefig(PATH)

        plt.figure()
        plt.hist(self.knee_action)
        plt.title("Knee action across episode")
        PATH = self.dir + self.name + "_knee_action_histogram.png"
        plt.savefig(PATH)

    def plot_temporal_action_change(self):
        if not os.path.isdir(self.dir):
            os.mkdir(self.dir)
        plt.figure()
        plt.title("Action Temporal Difference")
        plt.plot(range(len(self.hip_temporal_change)),self.hip_temporal_change,label='hip action change')
        plt.plot(range(len(self.knee_temporal_change)),self.knee_temporal_change,label='knee action change')
        plt.legend()
        PATH = self.dir + self.name + "action_td.png"
        plt.savefig(PATH)
