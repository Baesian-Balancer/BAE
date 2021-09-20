import math
import torch
import numpy as np
import scipy.stats
from torch._C import device
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

class StateTransitionNet(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_outputs, learning_rate, device):
            super(StateTransitionNet, self).__init__()
            self.num_outputs = num_outputs
            self.fc1 = nn.Linear(num_inputs, hidden_size)
            self.fc2 = nn.Linear(hidden_size, num_outputs)

            # self.l2_objective = nn.MSELoss()
            self.l1_objective = nn.L1Loss()
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
            self.device = device
    
    def forward(self, state_action):
        x = F.relu(self.fc1(state_action))
        x = torch.sigmoid(self.fc2(x)) # output normalized state values in [0,1]
        return x
    
    def get_state(self, state_actions):
        state_actions = torch.from_numpy(state_actions).float().unsqueeze(0)
        state_actions = state_actions.to(self.device)
        pred_state = self.forward(state_actions)
        return pred_state

    def update_net(self, network, pred_states, new_states):
        """ Calculate L2/L1 loss between predicted state """
        # torch.stack turns a list of tensors into a tensor
        pred_states = torch.stack(pred_states)
        new_states = torch.stack(new_states)
        pred_states = torch.squeeze(pred_states)
        
        # loss = self.l2_objective(pred_states.float(), new_states.float())
        loss = self.l1_objective(pred_states, new_states)

        network.optimizer.zero_grad()
        loss.backward()
        network.optimizer.step()