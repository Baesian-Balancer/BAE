import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
import copy, random, math
import time
import functools
from gym_ignition.utils import logger

from igBIBLE.MBRL.PolicyNet import PolicyNetwork
from igBIBLE.MBRL.StateTransitionNet import StateTransitionNet
from igBIBLE.igBIBLE import randomizers
from sklearn.metrics import mean_squared_error

# Set verbosity
logger.set_level(gym.logger.ERROR)
# logger.set_level(gym.logger.DEBUG)

device = torch.device("cpu")
# Available tasks
env_id = "Monopod-Gazebo-v1"


def make_env_from_id(env_id: str, **kwargs) -> gym.Env:
    import gym
    import igBIBLE
    return gym.make(env_id, **kwargs)


# Create a partial function passing the environment id
make_env = functools.partial(make_env_from_id, env_id=env_id)

# Wrap the environment with the randomizer.
# This is a simple example no randomization are applied.

env = randomizers.monopod_no_rand.MonopodEnvNoRandomizations(env=make_env)

# Wrap the environment with the randomizer.
# This is a complex example that randomizes both the physics and the model.
# env = randomizers.monopod.MonopodEnvRandomizer(
#     env=make_env, seed=42, num_physics_rollouts=5)

# Enable the rendering
# env.render('human')
# Initialize the seed
env.seed(42)


# Initialize state-transition network
HIDDEN_SIZE = 256
OBS_SHAPE = env.observation_space.shape[0]
ACTION_SHAPE = env.action_space.shape[0]
LEARNING_RATE = 1e-3

# Adding tuples appends instead of element-wise adding
INPUT_SHAPE = OBS_SHAPE + ACTION_SHAPE

state_transition_net = StateTransitionNet(INPUT_SHAPE, HIDDEN_SIZE, OBS_SHAPE, LEARNING_RATE, device)

# Training params
NUM_EPISODES = 500
MAX_STEPS = 1000
MEMORY_SIZE = 1000 # Remember previous MEMORY_SIZE steps

# List to hold (state_action, new_state) pairs in memory. #TODO: convert this to numpy
memory_list = []


def normalize_inputs(observations):
    observation_normed = np.zeros(OBS_SHAPE)
    for i, obs in enumerate(observations):
        # Get obs boundaries
        max_obs = env.observation_space.high[i]
        min_obs = env.observation_space.low[i]

        # Maxmin normalize observations
        obs_normed = (obs - min_obs) / (max_obs - min_obs)
        observation_normed[i] = obs_normed
    return observation_normed


def calculate_loss(pred_state, new_state):
    with torch.no_grad():
        pred_state.numpy()
        pred = np.squeeze(pred_state)
        loss = mean_squared_error(pred, new_state)
    return loss


def main():
    episode_rewards = []
    for episode in range(NUM_EPISODES):
        done = False
        obs = env.reset()
        obs = normalize_inputs(obs)

        # Setup
        pred_states = []
        new_states = []
        losses = []

        step = 0               
        while not done:
            actions = env.action_space.sample()

            # Get predicted state
            state_action = np.concatenate((obs,actions))
            pred_state = state_transition_net.get_state(state_action)

            # Take action, get next state
            new_obs, reward, done, _ = env.step(actions)
            new_obs = normalize_inputs(new_obs)

            # Save pred state and new state
            pred_states.append(pred_state)
            new_states.append(torch.from_numpy(new_obs))

            # Calculate loss
            loss = calculate_loss(pred_state, new_obs)
            losses.append(loss)

            step += 1
            if done is not True:
                done = True if step == MAX_STEPS else False
            if done:
                state_transition_net.update_net(state_transition_net, pred_states, new_states)

            obs = new_obs

        if episode % 5 == 0 and episode != 0:
            print("Episode {} - Last-5 Average: {}".format(episode,np.mean(losses[episode-5:episode])))
   
    # Plotting stuffs
    mean_losses = np.zeros(NUM_EPISODES)
    for epi in range(NUM_EPISODES):
        mean_losses[epi] = np.mean(losses[max(0,epi-50):(epi+1)])
    plt.plot(mean_losses)
    plt.show()

if __name__ == "__main__":
    main()