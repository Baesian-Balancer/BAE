import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
import copy, random, math
import time
import functools
from gym_ignition.utils import logger
from PolicyNet import PolicyNetwork
from gym_bb import randomizers
import timeit
# Set verbosity
logger.set_level(gym.logger.ERROR)
# logger.set_level(gym.logger.DEBUG)

# Available tasks
env_id = "Monopod-Gazebo-v2"

def make_env_from_id(env_id: str, **kwargs) -> gym.Env:
    import gym
    import gym_bb
    return gym.make(env_id, **kwargs)

# Create a partial function passing the environment id
make_env = functools.partial(make_env_from_id, env_id=env_id)

# Wrap the environment with the randomizer.
# This is a simple example no randomization are applied.
env = randomizers.monopod_no_rand.MonopodEnvNoRandomizations(env=make_env)

# Enable the rendering
# env.render('human')
# Initialize the seed
env.seed(69)

# Initialize policy network
HIDDEN_SIZE = 2048
OBS_SHAPE = env.observation_space.shape
OBS_SIZE = OBS_SHAPE[0]
ACTION_SIZE = 2
LR = 1e-4
GAMMA = 0.9
policy_net = PolicyNetwork(OBS_SIZE, HIDDEN_SIZE, ACTION_SIZE, LR, GAMMA, torch.device("cpu"))

# Training params
NUM_EPISODES = 200
MAX_STEPS = 3000
# SHOW_EVERY = 10
LOG_EVERY = 20

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

    
def main():
    episode_rewards = []
    for episode in range(NUM_EPISODES):
        done = False
        obs = env.reset()

        obs = normalize_inputs(obs)

        # Setup
        log_probs = []
        rewards = []
        episode_reward = 0
        step = 0
        # start_time = timeit.default_timer()       
        while not done:
            actions, log_prob = policy_net.get_action(obs)
            new_obs, reward, done, _ = env.step(actions)
            new_obs = normalize_inputs(new_obs)

            log_probs.append(log_prob)
            rewards.append(reward)
            episode_reward += reward
            step += 1

            if done is not True:
                done = True if step == MAX_STEPS else False

            if done:
                policy_net.update_policy(policy_net, rewards, log_probs)
                episode_rewards.append(episode_reward)

            obs = new_obs

            if step == 2999:
                print('ALL DONE')
        # print(f"Episode took {timeit.default_timer() - start_time}")
        if episode % LOG_EVERY == 0 and episode != 0:
            # print(f"Last Episode took: {time.time() - start} Steps: {step}")
            print(f"Episode {episode} - Last-{LOG_EVERY} Average: {np.mean(episode_rewards[episode-LOG_EVERY:episode])}")
    # Plotting stuffs
    mean_rewards = np.zeros(NUM_EPISODES)
    for epi in range(NUM_EPISODES):
        mean_rewards[epi] = np.mean(episode_rewards[max(0,epi-50):(epi+1)])
    plt.plot(mean_rewards)
    plt.show()

if __name__ == "__main__":
    main()