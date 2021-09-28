from MBRL_utils import look_ahead
import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import functools
from gym_ignition.utils import logger
from BB_gym_Envs import randomizers

from PolicyNet import PolicyNetwork
from StateTransitionNet import StateTransitionNet

from sklearn.metrics import mean_squared_error

# Set verbosity
logger.set_level(gym.logger.ERROR)
# logger.set_level(gym.logger.DEBUG)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")
print(device)
# Available tasks
env_id = "Monopod-Gazebo-v2"

def make_env_from_id(env_id: str, **kwargs) -> gym.Env:
    import gym
    import BB_gym_Envs
    return gym.make(env_id, **kwargs)

# Create a partial function passing the environment id
make_env = functools.partial(make_env_from_id, env_id=env_id)

# Wrap the environment with the randomizer.
# This is a simple example no randomization are applied.
env = randomizers.monopod_no_rand.MonopodEnvNoRandomizations(env=make_env)

# Initialize the seed
env.seed(69)
# env.render('human')

# Initialize networks
PN_HIDDENSIZE = 2048
ST_HIDDENSIZE = 256
OBS_SIZE = env.observation_space.shape[0]
ACTION_SIZE = env.action_space.shape[0]
STATE_INPUT_SIZE = OBS_SIZE + ACTION_SIZE
PN_LR = 1e-4
ST_LR = 1e-3
GAMMA = 0.95


policy_net = PolicyNetwork(OBS_SIZE, PN_HIDDENSIZE, ACTION_SIZE, PN_LR, GAMMA, device)
if os.path.exists("/home/keithg/Baesian/transition_net.pt"):
    print("Loaded State-Transition Network")
    transition_net = torch.load("/home/keithg/Baesian/transition_net.pt")
else:
    transition_net = StateTransitionNet(STATE_INPUT_SIZE, ST_HIDDENSIZE, OBS_SIZE, ST_LR, device)

policy_net.to(device)
transition_net.to(device)

# Training params
NUM_EPISODES = 100
MAX_STEPS = 3000
ROLLOUT_EVERY = 10
UPDATE_EVERY = 100

# How often to print the average of last LOG_EVERY episodes
LOG_EVERY = 20

# Look ahead setup
LOOKAHEAD_NUMACTIONS = 4
LOOKAHEAD_STEPS = 3

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


def calculate_loss(pred_state, new_state):
    with torch.no_grad():
        pred = np.squeeze(pred_state.cpu())
        loss = mean_squared_error(pred, new_state)
    return loss


# def test_rollouts():
#     steps = 100
#     episodes = 10
#     generate_rollouts(env, policy_net, transition_net, steps, episodes)


def main():
    episode_rewards = []
    episode_losses = []
    time_count = 0
    for episode in range(NUM_EPISODES):
        start = time.time()
        done = False
        obs = env.reset()
        obs = normalize_inputs(obs)

        # # Perform rollouts. 
        # if episode >= 100 and episode % ROLLOUT_EVERY == 0:
        #     generate_rollouts(env, policy_net, transition_net, 500, 10)

        # PN Setup
        log_probs = []
        rewards = []
        episode_reward = 0
        
        # ST Setup
        pred_states = []
        new_states = []
        losses = []
        episode_loss = 0

        step = 0
        while not done:
            # Get actions and log probs from policy net
            # best_action, log_prob = policy_net.get_action(obs)
            best_action, log_prob = look_ahead(env, obs, policy_net, transition_net, LOOKAHEAD_NUMACTIONS, LOOKAHEAD_STEPS)
            # Get predicted state
            state_action = np.concatenate((obs,best_action))
            pred_state = transition_net.get_state(state_action)

            # Take action, get next state
            new_obs, reward, done, _ = env.step(best_action)
            new_obs = normalize_inputs(new_obs)

            # ST - Save pred state and new state
            pred_states.append(pred_state)
            new_states.append(torch.from_numpy(new_obs).to(device))

            # PN - Save log_probs and rewards
            log_probs.append(log_prob)
            rewards.append(reward)

            # ST - Calculate loss
            loss = calculate_loss(pred_state, new_obs)
            losses.append(loss)

            episode_reward += reward
            episode_loss += loss
            
            step += 1
            if done is not True:
                done = True if step == MAX_STEPS else False
            if done:
                # if step > 400:
                #     rewards = [rew + step for rew in rewards]
                policy_net.update_policy(policy_net, rewards, log_probs)
                # transition_net.update_net(transition_net, pred_states, new_states)
                episode_rewards.append(episode_reward)
                episode_losses.append(episode_loss)
            elif step % UPDATE_EVERY == 0:
                policy_net.update_policy(policy_net, rewards, log_probs)
                # transition_net.update_net(transition_net, pred_states, new_states)
                rewards = []
                log_probs = []
            obs = new_obs
            if step == 2999:
                print('ALL DONE')
        if episode % LOG_EVERY == 0 and episode != 0:
            # print(f"Last Episode took: {time.time() - start} Steps: {step}")
            print(f"Episode {episode} - Last-{LOG_EVERY} Average: {np.mean(episode_rewards[episode-LOG_EVERY:episode])}")
            print(f"Episode {episode} - Last-{LOG_EVERY} Average: {np.mean(losses[episode-LOG_EVERY:episode])}")
    
    # # Save stuffs
    torch.save(transition_net, "transition_net.pt")

    # Plotting stuffs
    mean_rewards = np.zeros(NUM_EPISODES)
    mean_losses = np.zeros(NUM_EPISODES)
    for epi in range(NUM_EPISODES):
        mean_rewards[epi] = np.mean(episode_rewards[max(0,epi-50):(epi+1)])
        mean_losses[epi] = np.mean(episode_losses[max(0,epi-50):(epi+1)])

    plt.subplot(1, 2, 1)
    plt.plot(mean_rewards)
    plt.ylabel("rewards")
    # plt.show()
    plt.subplot(1, 2, 2)
    plt.plot(mean_losses)
    plt.ylabel("loss")
    plt.show()

    # # Plotting stuffs
    # moving_avg_reward = np.convolve(episode_rewards, np.ones((LOG_EVERY,)) / LOG_EVERY, mode="valid")
    # moving_avg_loss = np.convolve(episode_losses, np.ones((LOG_EVERY,)) / LOG_EVERY, mode="valid")

    # plt.subplot(1, 2, 1)
    # plt.plot([i for i in range(len(moving_avg_reward))], moving_avg_reward)
    # plt.ylabel("rewards")
    # # plt.show()
    # plt.subplot(1, 2, 2)
    # plt.plot([i for i in range(len(moving_avg_loss))], moving_avg_loss)
    # plt.ylabel("loss")
    # plt.show()

if __name__ == "__main__":
    main()
    # test_rollouts()