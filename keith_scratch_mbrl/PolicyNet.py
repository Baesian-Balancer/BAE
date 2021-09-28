import torch
import numpy as np
import scipy.stats
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.autograd import Variable
from numba import jit
from numba.typed import List

@jit
def calculate_discounted_reward(rewards, gamma):
    """
    Calculate future discounted reward for each time step, using
    formula G_t = sum [gamma^(t'-t-1) * r_t']
    """
    discounted_rewards = np.zeros(len(rewards))
    for t in range(len(rewards)):
        G_t = 0 
        discount_power = 0
        for r in rewards[t:]:
            G_t += (gamma**discount_power)*r
            discount_power += 1
        discounted_rewards[t] = G_t

    # Standardize rewards.
    # Note: setting the unbiased = False flag means .std() will return 0 for cases where there is a single reward.
    # Returns NaN otherwise...that was fun to figure out.
    # discounted_rewards = ((discounted_rewards - discounted_rewards.mean()) / 
    #                     (discounted_rewards.std(unbiased = False) + 1e-9))
    discounted_rewards = ((discounted_rewards - discounted_rewards.mean()) / 
                    (discounted_rewards.std() + 1e-9))
    return discounted_rewards

def convert_to_numba_list(rewards):
    return List(rewards)

# Using two fully-connected linear layers for NN to output mean, std describing normal
# distribution over actions.
class PolicyNetwork(nn.Module):
    # SIGMA = 0.2
    def __init__(self, num_inputs, hidden_size, num_outputs, learning_rate, gamma, device):
        super(PolicyNetwork, self).__init__()
        self.num_outputs = num_outputs
        self.gamma = gamma
        self.policy_net = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Tanh()
        )
        self.logstd = nn.Parameter(torch.ones(2, ), requires_grad=True)

        # self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # self.scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 200)
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
        self.device = device
    
    def forward(self, state):
        mean = self.policy_net(state)
        return mean, self.logstd
    
    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        mean, logstd = self.forward(Variable(state))
        # mean = self.forward(Variable(state))
        actions = np.zeros((self.num_outputs,))
        log_probs = np.zeros((self.num_outputs,))
        with torch.no_grad():
            mean = mean.cpu()
            logstd = logstd.cpu()
            for i in range(self.num_outputs):
                # Sample actions from normal distributions
                std = np.exp(logstd[i])
                gaussian = Normal(mean[0][i], std)
                action = gaussian.sample((1,)).clamp(-1, 1)

                log_prob = gaussian.log_prob(action)
                actions[i] = action
                log_probs[i] = log_prob
        return actions, log_probs

    def update_policy(self, policy_network, rewards, log_probs):
        # Get discounted reward
        rewards = convert_to_numba_list(rewards)
        discounted_rewards = calculate_discounted_reward(rewards, self.gamma)
        discounted_rewards = torch.tensor(discounted_rewards, device = self.device)

        # Calculate loss and place in policy_gradient list
        policy_gradient = []
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

        policy_network.optimizer.step() # updates using gradient calculated above

        # # # UNCOMMENT FOR SCHEDULER # # # 
        # policy_network.scheduler.step()

