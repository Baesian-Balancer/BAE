import math
import torch
import numpy as np
import scipy.stats
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# Using two fully-connected linear layers for NN to output mean, std describing normal
# distribution over actions.
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_outputs, learning_rate, gamma, device):
        super(PolicyNetwork, self).__init__()
        self.num_outputs = num_outputs
        self.gamma = gamma
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_outputs)

        self.logstd = nn.Parameter(torch.ones(2, ), requires_grad=True)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        mean = torch.tanh(self.fc2(x)) # normalize mean to [-1, 1]

        return mean, self.logstd
    
    def get_action(self,state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        mean, logstd = self.forward(Variable(state))

        actions = np.zeros((self.num_outputs,))
        log_probs = np.zeros((self.num_outputs,))
        with torch.no_grad():
            # Sample actions from normal distributions

            for i in range(self.num_outputs):
                # Sample actions from normal distributions
                std = np.exp(logstd[i])
                gaussian = scipy.stats.norm(loc=mean[0,i], scale=std)
                
                action = gaussian.rvs()
                prob = gaussian.pdf(action)
                log_prob = torch.log(torch.Tensor([prob]))
                
                actions[i] = action
                log_probs[i] = log_prob

        return actions, log_probs

    def update_policy(self, policy_network, rewards, log_probs):
        
        # Calculate future discounted reward at each time step.
        # G_t = sum_t+1=>T gamma^(t'-t-1)*r_t'
        discounted_rewards = []
        for t in range(len(rewards)):
            G_t = 0 
            discount_power = 0
            for r in rewards[t:]:
                G_t += (self.gamma**discount_power)*r
                discount_power += 1
            discounted_rewards.append(G_t)
        discounted_rewards = torch.tensor(discounted_rewards)

        # Standardize rewards.
        # Note: setting the unbiased = False flag means .std() will return 0 for cases where there is a single reward.
        # Returns NaN otherwise...that was fun to figure out.
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std(unbiased = False) + 1e-9)
        
        policy_gradient = []
        
        # Calculate loss. Summing each action's negative log-likelihoods
        for log_prob, G_t in zip(log_probs, discounted_rewards):
            log_prob1 = torch.tensor(log_prob[0])
            log_prob2 = torch.tensor(log_prob[1])
            log_prob1.requires_grad = True
            log_prob2.requires_grad = True

            neg_LL = -(log_prob1 + log_prob2) * G_t / 2
            policy_gradient.append(neg_LL)
    
        # Update 
        policy_network.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()

        policy_network.optimizer.step() # updates value using gradient calculated above

        # # # UNCOMMENT FOR SCHEDULER # # # 
        # policy_network.scheduler.step() # Uncomment if using learning rate scheduler

