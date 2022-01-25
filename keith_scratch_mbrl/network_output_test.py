from PolicyNet import PolicyNetwork
import torch


# Load Model
OBS_SIZE = 8
ACTION_SIZE = 2
PN_LR = 1e-4
ST_LR = 1e-3
GAMMA = 0.95
PN_HIDDENSIZE = 256
device = "cpu"
policy_net = PolicyNetwork(OBS_SIZE, PN_HIDDENSIZE, ACTION_SIZE, PN_LR, GAMMA, device)

model_file = "/home/capstone/capstone/rl-algorithm-exploration/policy_net.pt"

policy_net.load_state_dict(torch.load(model_file))

test_data = torch.ones((1, 8))

print(policy_net.forward(test_data))
