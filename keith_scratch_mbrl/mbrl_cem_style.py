from igBIBLE.MBRL.MBRL_utils import look_ahead
import torch
import numpy as np
import gym
import numpy as np
import matplotlib.pyplot as plt
import os, time, functools
import heapq
from gym_ignition.utils import logger
from torch.backends.mkl import is_available
# from igBIBLE.CEM.PolicyNetwork_CEM import PolicyNetwork
from igBIBLE.igBIBLE import randomizers
from PolicyNet import PolicyNetwork
from StateTransitionNet import StateTransitionNet

# Set verbosity
logger.set_level(gym.logger.ERROR)
# logger.set_level(gym.logger.DEBUG)

env_id = "Monopod-Gazebo-v2"

def make_env_from_id(env_id: str, **kwargs) -> gym.Env:
    import gym
    import igBIBLE
    return gym.make(env_id, **kwargs)

# Create a partial function passing the environment id
make_env = functools.partial(make_env_from_id, env_id=env_id)

# Wrap the environment with the randomizer.
# This is a simple example no randomization are applied.
env = randomizers.monopod_no_rand.MonopodEnvNoRandomizations(env=make_env)

# Initialize the seed
env.seed(69)
# env.render('human')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")
print(device)

# Initialize networks
PN_HIDDENSIZE = 2048
OBS_SIZE = env.observation_space.shape[0]
ACTION_SIZE = env.action_space.shape[0]
STATE_INPUT_SIZE = OBS_SIZE + ACTION_SIZE
PN_LR = 1e-4
ST_LR = 1e-3
ST_HIDDENSIZE = 256
GAMMA = 0.95
policy_net = PolicyNetwork(OBS_SIZE, PN_HIDDENSIZE, ACTION_SIZE, PN_LR, GAMMA, device)
if os.path.exists("/home/keithg/Baesian/transition_net.pt"):
    print("Loaded State-Transition Network")
    transition_net = torch.load("/home/keithg/Baesian/transition_net.pt")
else:
    transition_net = StateTransitionNet(STATE_INPUT_SIZE, ST_HIDDENSIZE, OBS_SIZE, ST_LR, device)
policy_net.to(device)
transition_net.to(device)
# Training Params 
MAX_STEPS = 2000
TOP_PERCENTILE = 0.8
NUM_EPOCHS = 100
NUM_TRIALS = 20

# Look ahead setup
LA_actions = 3
LA_steps = 3

def normalize_inputs(observations):
    observation_normed = np.zeros(OBS_SIZE)
    for i, obs in enumerate(observations):
        # Get obs boundaries
        max_obs = env.observation_space.high[i]
        min_obs = env.observation_space.low[i]

        # Maxmin normalize observations
        obs_normed = (obs - min_obs) / (max_obs - min_obs)
        observation_normed[i] = obs_normed
    return observation_normed

# generate a list of observation/action pairs for n = num_of_trials episodes. Save total reward for each episode to train nn
def generate_trials(env, policy_net, num_trials, max_steps):
    episode_rewards = []
    episode_log_probs = []
    episode_total_reward = []

    for episode in range(num_trials):
        rewards = []
        log_probs = []
        total_reward = 0
        done = False
        new_obs = env.reset()
        new_obs = normalize_inputs(new_obs)

        step = 0 
        while not done:
            actions, log_prob = look_ahead(env, new_obs, policy_net, transition_net, LA_actions, LA_steps)

            new_obs, reward, done, _ = env.step(actions)
            new_obs = normalize_inputs(new_obs)
            
            rewards.append(reward)
            log_probs.append(log_prob)     
            total_reward += reward

            step += 1
            if done is not True:
                done = True if step == MAX_STEPS else False
            if done:
                episode_rewards.append(rewards)
                episode_log_probs.append(log_probs)
                episode_total_reward.append(total_reward)
                break
            if step == 1999:
                print("FINAL STEP")
    return episode_rewards, episode_log_probs, episode_total_reward

# want to use the best episodes to learn
def top_episodes(ep_rewards, ep_log_probs, ep_r_total, percentile):
    top_rewards = []
    top_log_probs = []
    # how many best episodes we take
    n = round(len(ep_r_total)*(1-percentile))
    if n < 1:
        n = 1

    # this guy works in linear time to find indices of n biggest elements. By using range(len()) and .take
    # it returns indices instead of the actual highest values
    best_episode_indices = heapq.nlargest(n, range(len(ep_r_total)),ep_r_total.__getitem__)

    for index in best_episode_indices:
        top_rewards.extend(ep_rewards[index])
        top_log_probs.extend(ep_log_probs[index])
    return top_rewards, top_log_probs

def main():
    for epoch in range(NUM_EPOCHS):
        print(f"On Epoch: {epoch}")
        trial_obs, trial_actions, trial_rewards = generate_trials(env, policy_net, NUM_TRIALS, MAX_STEPS)
        print(np.mean(trial_rewards))
        best_rewards, best_log_probs = top_episodes(trial_obs, trial_actions, trial_rewards, TOP_PERCENTILE)
        policy_net.update_policy(policy_net, best_rewards, best_log_probs)

if __name__ == "__main__":
    main()