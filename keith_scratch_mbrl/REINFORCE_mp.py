import torch
import gym
import time
import functools
import numpy as np
from gym_ignition.utils import logger
from gym_bb import randomizers
from gym_bb.common.mp_env import make_mp_envs
from gym_bb.monitor.monitor import VecMonitor, VecMonitorPlot
from PolicyNet import PolicyNetwork
import multiprocessing
import os
import sys


# Set verbosity
logger.set_level(gym.logger.ERROR)
# logger.set_level(gym.logger.DEBUG)

def normalize_envs_inputs(envs, observation_arr):
	for i, observation in enumerate(observation_arr):
		observation_normed = np.zeros(OBS_SIZE)
		for j, obs in enumerate(observation):
			# Get obs boundaries
			max_obs = envs.observation_space.high[j]
			min_obs = envs.observation_space.low[j]

			# Maxmin normalize observations
			obs_normed = (obs - min_obs) / (max_obs - min_obs)
			observation_normed[j] = obs_normed
		observation_arr[i] = observation_normed
	return observation_arr

def main_loop(envs):
	start = time.time()  

	# Enable the rendering
	# env.render('human')
	current_cumulative_rewards = np.zeros(NUM_ENVS)

	obs_arr = envs.reset()
	obs_arr = normalize_envs_inputs(envs, obs_arr)

	# TODO: Tracking of rewards and log probs initialized here
	# These lists hold the rew and log prob arrays for each time step
	rewards = []
	log_probs = []
	for step in range(NUMBER_TIME_STEPS):
		# Get actions and log probs from the policy net
		policy_net_preds_arr = [policy_net.get_action(obs) for obs in obs_arr]
		actions_arr = np.asarray([pred[0] for pred in policy_net_preds_arr])
		log_probs_arr = np.asarray([pred[1] for pred in policy_net_preds_arr])

		new_obs_arr, reward_arr, done_arr, _ = envs.step(actions_arr)
		new_obs_arr = normalize_envs_inputs(envs, new_obs_arr)

		rewards.append(reward_arr)
		log_probs.append(log_probs_arr)
		current_cumulative_rewards += reward_arr

		# TODO: Optimize/streamline this? Maybe an envs.update method to do it for all envs that are done? Or
		# class to handle data and parse and filter to pass to models.
		if any(done_arr):
			for i, done in enumerate(done_arr):
				if done:
					env_rewards = [rew[i] for rew in rewards]
					env_log_probs = [log_prob[i] for log_prob in log_probs]
					policy_net.update_policy(policy_net, env_rewards, env_log_probs)	
			
			# print('rollout info: ', envs.do_rollout(observation_arr))
			current_cumulative_rewards[done_arr] = 0
		obs_arr = new_obs_arr
	
	print(f"Total time elapsed is {time.time() - start}")
	envs.close()
	time.sleep(5)


# if __name__ == '__main__':
# 	try:
# 		device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
		
# 		# Available tasks
# 		env_id = "Monopod-Gazebo-v1"
# 		NUM_ENVS = multiprocessing.cpu_count()
# 		NUMBER_TIME_STEPS = 10000
# 		seed = 69

# 		fenvs = make_mp_envs(env_id, NUM_ENVS, seed,
# 							 randomizers.monopod.MonopodEnvRandomizer)
# 		# envs = VecMonitor(envs)
# 		envs = VecMonitorPlot(
# 			fenvs, plot_path=os.path.expanduser('~')+'/Desktop/plot')
		
# 		# Initialize networks
# 		PN_HIDDENSIZE = 2048
# 		OBS_SIZE = envs.observation_space.shape[0]
# 		ACTION_SIZE = envs.action_space.shape[0]
# 		print(OBS_SIZE, ACTION_SIZE)
# 		PN_LR = 1e-4
# 		GAMMA = 0.95
# 		policy_net = PolicyNetwork(OBS_SIZE, PN_HIDDENSIZE, ACTION_SIZE, PN_LR, GAMMA, device)

# 		envs.reset()
# 		main_loop(envs)
# 	except Exception as e:
# 		print(e)
# 		try:
# 			try:
# 				envs.close()
# 			except Exception as e:
# 				pass
# 			sys.exit(0)
# 		except SystemExit:
# 			os._exit(0)

if __name__ == '__main__':
	# try:
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	
	# Available tasks
	env_id = "Monopod-Gazebo-fh-fby-v1"
	NUM_ENVS = multiprocessing.cpu_count()
	NUM_ENVS = 2
	NUMBER_TIME_STEPS = 50_000
	seed = 69

	fenvs = make_mp_envs(env_id, NUM_ENVS, seed,
							randomizers.monopod.MonopodEnvRandomizer)
	# envs = VecMonitor(envs)


	# Set episodes_for_refresh and save_every_num_episodes.
	envs = VecMonitorPlot(
		fenvs, plot_path=os.path.expanduser('~')+'/Desktop/plot')
	
	# Initialize networks
	PN_HIDDENSIZE = 2048
	OBS_SIZE = envs.observation_space.shape[0]
	ACTION_SIZE = envs.action_space.shape[0]
	PN_LR = 1e-5
	GAMMA = 0.95
	policy_net = PolicyNetwork(OBS_SIZE, PN_HIDDENSIZE, ACTION_SIZE, PN_LR, GAMMA, device)
	policy_net.to(device)

	start = time.time()  
	main_loop(envs)
	print(time.time() - start)

	# except Exception as e:
	# 	print(e)
	# 	try:
	# 		try:
	# 			envs.close()
	# 		except Exception as e:
	# 			pass
	# 		sys.exit(0)
	# 	except SystemExit:
	# 		os._exit(0)