from ctypes import util
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
import utils 

class RandomNetworkDistillation(nn.Module):
    """
    Two neural networks predict next state. The error in the prediction
    is used as a measure of curiousity. Higher error means less familiar state,
    more curiousity.
    """
    def __init__(self,
                obs_dim: int,
                action_dim: int,
                hidden_dim = 256,
                hidden_depth = 1):
        super().__init__()

        self.target_trunk = utils.mlp(obs_dim + action_dim, hidden_dim, obs_dim, hidden_depth)
        self.pred_trunk = utils.mlp(obs_dim + action_dim, hidden_dim, obs_dim, hidden_depth)

        self.loss = F.mse_loss
        self.stats = RunningStats()


    def forward(self, obs, action):
        """ Input (obs, action) through networks and return difference """
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        target = self.target_trunk(obs_action)
        pred = self.pred_trunk(obs_action)
        return self.loss(pred, target)

    def get_curiosity(self, obs, action):
        curiosity = self.forward(obs, action)
        # return (curiosity - self.stats.mean()) / self.stats.standard_deviation()
        return curiosity

    def update_stats(self, curiosity):
        self.stats.push(curiosity)
    
    def normalize(self, curiosity):
        return (curiosity - self.stats.mean()) / self.stats.standard_deviation()


# https://stackoverflow.com/questions/1174984/how-to-efficiently-calculate-a-running-standard-deviation
class RunningStats:
    """ Welford's algorithm for running mean/std """
    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0
        self.epsilon = 1e-4
    def clear(self):
        self.n = 0
    
    def push(self, x):
        self.n += 1
    
        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)
        
            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0
    
    def standard_deviation(self):
        # print(math.sqrt(self.variance()) + self.epsilon)
        return math.sqrt(self.variance()) + self.epsilon