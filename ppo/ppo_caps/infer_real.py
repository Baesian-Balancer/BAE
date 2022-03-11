import wandb

import numpy as np
import torch
from tqdm.auto import tqdm
from torch.optim import Adam
import gym
import gym_os2r_real
import time
import core
import functools
import os
import sys
sys.path.append("../../")
from plotting import PlotUtils
# from spinup.utils.logx import EpochLogger
# from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
# from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


def make_env(env_id):
    env = gym.make(env_id)
    return env


def ppo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000000,
        target_kl=0.01, save_freq=10):

    # Random seed
    seed += 10000 #* proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()

    # env.render()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    path = './real_exp/2022_03_08_02_55_08'
    name = 'best_model_step_819999'
    checkpoint = torch.load(f'{path}/{name}.pt')
    ac.load_state_dict(checkpoint['actor_state_dict'])

    plotting = PlotUtils(name, f'{path}/plots/{name}/')

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

    # Set up model saving
    # logger.setup_pytorch_saver(ac)

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    smoothed_act = 0
    # Main loop: collect experience in env and update/log each epoch
    progress_bar = tqdm(range(epochs),desc='Epoch')
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            a = ac.step(torch.as_tensor(o, dtype=torch.float32), eval=True)

            # plotting.add_action(a)
            smoothed_act = 0.33*smoothed_act + 0.66*a
            smoothed_act = a
            o, r, d, _ = env.step(smoothed_act)
            ep_ret += r
            ep_len += 1

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if terminal:
                #     # only save EpRet / EpLen if trajectory finished
                #     # logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0
        progress_bar.update(1)
        # if (epoch % save_freq == 0) or (epoch == epochs-1):
        #     logger.save_state({'env': env}, None)

        plotting.plot_temporal_action_change()
        plotting.plot_action_histogram()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Real-monopod-balance-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--exp_name', type=str, default='ppo real')
    args = parser.parse_args()

    # mpi_fork(args.cpu)  # run parallel code with mpi

    # from spinup.utils.run_utils import setup_logger_kwargs
    # logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ppo(lambda : make_env(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs, max_ep_len=args.steps)
