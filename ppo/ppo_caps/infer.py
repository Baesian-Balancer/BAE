import wandb

import numpy as np
import torch
from tqdm.auto import tqdm
from torch.optim import Adam
import gym
import gym_os2r
import time
import core
import functools
import os
from gym_os2r import randomizers
import sys
sys.path.append("../../")
from plotting import PlotUtils
def make_env(env_id,**kwargs):
    def make_env_from_id(env_id: str, **kwargs) -> gym.Env:
        return gym.make(env_id, **kwargs)

    # Create a partial function passing the environment id
    create_env = functools.partial(make_env_from_id, env_id=env_id,**kwargs)
    #env = randomizers.monopod_no_rand.MonopodEnvNoRandomizer(env=create_env)
    env = randomizers.monopod.MonopodEnvRandomizer(env=create_env)

    # Enable the rendering
    env.render('human')

    # Initialize the seed
    print(env)
    time.sleep(2)
    return env

def ppo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000000,
        target_kl=0.01, save_freq=10):

    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()

    # env.render()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    name = "best_model_step_503999"
    path = "/home/nickioan/capstone/cap_repos/models/2022_04_01_03_03_00/"
    checkpoint = torch.load(path + name + ".pt")
    ac.load_state_dict(checkpoint['actor_state_dict'])

    plotting = PlotUtils(name, f'{path}plots/{name}/' )
    # Sync params across processes
    # sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    # logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch )#/ num_procs())

    # Set up function for computing PPO policy loss


    # Set up function for computing value loss


    # Set up optimizers for policy and value function

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    progress_bar = tqdm(range(epochs),desc='Epoch')
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):

            # if np.random.rand() < 0.30:
            #     o += np.random.uniform(-0.05,0.05,np.size(o))
            o[4] = o[4]*0.5
            a = ac.step(torch.as_tensor(o, dtype=torch.float32),deterministic=True)
            plotting.add_action(a)

            next_o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            o =  o + (np.random.uniform() > 0.15)*np.random.normal(np.zeros(np.shape(o)), 0.2)

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: Evaluation trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                o, ep_ret, ep_len = env.reset(), 0, 0
        progress_bar.update(1)

    plotting.plot_temporal_action_change()
    plotting.plot_action_histogram()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Monopod-nonorm-balance-v3')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=10000)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--exp_name', type=str, default='ppo caps')
    args = parser.parse_args()

    env_kwargs = {'task_mode': 'fixed_hip'}

    ppo(lambda : make_env(args.env,**env_kwargs), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs, max_ep_len=args.steps)
