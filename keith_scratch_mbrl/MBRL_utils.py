import numpy as np
import copy

def normalize_inputs(env, observations, OBS_SIZE):
    observation_normed = np.zeros(OBS_SIZE)
    for i, obs in enumerate(observations):
        # Get obs boundaries
        max_obs = env.observation_space.high[i]
        min_obs = env.observation_space.low[i]

        # Maxmin normalize observations
        obs_normed = (obs - min_obs) / (max_obs - min_obs)
        observation_normed[i] = obs_normed
    return observation_normed

def generate_rollouts(env, policy_net, transition_net, steps, episodes, MAX_STEPS):
    """
    Perform simple trajectory rollouts, ie, run episodes in our imagination;  use the
    state-transition and policy net to generate data, calculate reward, update policy net
    """
    print("Starting rollouts...")
    # Create imaginary copy of env
    imaginary_env = copy.copy(env)    

    episode_rewards = []
    # Go through episodes
    for episode in range(episodes):
        done = False

        # Get obs
        obs = imaginary_env.reset()
        obs = normalize_inputs(obs)

        # Policy Setup
        log_probs = []
        rewards = []
        episode_reward = 0

        step = 0
        while not done and step < steps:
            # Get predicted action and probability.
            obs = np.squeeze(obs)
            actions, log_prob = policy_net.get_action(obs)
            state_action = np.concatenate((obs,actions))
            
            # Get predicted state and calculate reward and done
            imaginary_env = imaginary_env.unwrapped
            new_obs = imaginary_env.state = transition_net.get_state(state_action).detach().cpu().numpy()
            reward = imaginary_env.task.get_reward()
            done = imaginary_env.task.is_done()
            
            # Store info
            log_probs.append(log_prob)
            rewards.append(reward)
            episode_reward += reward
            step += 1

            if not done:
                done = True if step == MAX_STEPS else False
            if done:
                policy_net.update_policy(policy_net, rewards, log_probs)
                episode_rewards.append(episode_reward)
            obs = new_obs

def look_ahead(env, obs, policy_net, transition_net, num_actions, steps):
    """
    Performs trajectory rollouts and returns the best action based on which initial
    action lead to largest reward
    """
    # Copy env and save initial state
    imaginary_env = copy.copy(env)
    initial_state = obs

    # Sample actions
    action_list = [policy_net.get_action(initial_state) for i in range(num_actions)]
    reward_list = []
    for action, _ in action_list:
        curr_state = initial_state.copy()

        state_action = np.concatenate((curr_state, action))
        total_reward = 0
        for step in range(steps):
            # Unwrap environment to access hidden class variables.
            # Then get state from transition_net, add reward to total
            imaginary_env.unwrapped
            # print(imaginary_env.__dict__)
            new_state = imaginary_env.state = transition_net.get_state(state_action).detach().cpu().numpy()
            reward = imaginary_env.task.get_reward()
            total_reward += reward
            done = imaginary_env.task.is_done()
            if done:
                break
            curr_state = new_state
        reward_list.append(total_reward)
    max_rew = max(reward_list)
    max_index = reward_list.index(max_rew)
    best_action, log_prob = action_list[max_index]
    return best_action, log_prob
        
