import numpy as np
import gym
import shutil
import time
import os

from rl_smoothness.algs.baseline_ppo.ppo import trainer

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

# from rl_smoothness.train_alg import TrainAlg

# class PPOB(TrainAlg):
    
#     def name(self):
#         return "ppob"

#     def add_args(self, parser):
#         parser.add_argument('--timesteps', '-ts', type=int, default=800000)
#         return parser

#     def train(self, args):
#         trainer.train(
#             env = args['env'], 
#             num_timesteps = args['timesteps'], 
#             seed = args['seed'], 
#             ckpt_dir = args['ckpt_dir'],
#             ckpt_freq = 50000, 
#             horizon = 512, 
#             optim_batchsize = 64,
#             optim_epochs = 5,
#             gamma = 0.9
#         )


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--exp_name', type=str, default='baselines_ppo')

    parser.add_argument('--lam_a', type=float, help='Regularization coeffecient on action smoothness (valid > 0)', default=-1.)
    parser.add_argument('--lam_s', type=float, help='Regularization coeffecient on state mapping smoothness (valid > 0)', default=-1.)
    parser.add_argument('--eps_s', type=float, help='Variance coeffecient on state mapping smoothness (valid > 0)', default=1.)
    parser.add_argument('--test', type=str, help='test the agent', default=None)
    args = parser.parse_args()

    # signal(SIGINT, handler)

    seed = args.seed
    training_dir = "ppo_baselines"
    print ("Storing results to ", training_dir)


    # Algorithm parameters
    # timesteps = 2000000
    ckpt_dir = os.path.join(training_dir, "checkpoints")
    render = False
    ckpt_freq = 1000
    schedule = "linear"
    step_size = 3e-4
    horizon = 2048
    batchsize = 32
    gamma = 0.99

    from rl_smoothness.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, seed)

    env = gym.make(args.env)
    env.seed(seed)

    env.noise_sigma = 1

    # from rl_smoothness.utils.run_utils import setup_logger_kwargs
    # logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    trainer.train(
        lambda: env, 
        seed, 
        None,
        ckpt_dir,
        render,
        None,
        ckpt_freq, 
        schedule = schedule, 
        optim_stepsize = step_size, 
        horizon = horizon, 
        optim_batchsize = batchsize,
        optim_epochs = 10,
        gamma = args.gamma,
        lam_a = args.lam_a,
        lam_s = args.lam_s,
        eps_s = args.eps_s,
        test = args.test,
        logger_kwargs = logger_kwargs,
        epochs = args.epochs
        )

# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--env', type=str, default='HalfCheetah-v2')
#     parser.add_argument('--hid', type=int, default=256)
#     parser.add_argument('--l', type=int, default=2)
#     parser.add_argument('--gamma', type=float, default=0.99)
#     parser.add_argument('--seed', '-s', type=int, default=0)
#     parser.add_argument('--epochs', type=int, default=50)
#     parser.add_argument('--exp_name', type=str, default='ddpg')

#     parser.add_argument('--lam_a', type=float, help='Regularization coeffecient on action smoothness (valid > 0)', default=-1.)
#     parser.add_argument('--lam_s', type=float, help='Regularization coeffecient on state mapping smoothness (valid > 0)', default=-1.)
#     parser.add_argument('--eps_s', type=float, help='Variance coeffecient on state mapping smoothness (valid > 0)', default=1.)
#     args = parser.parse_args()

#     from rl_smoothness.utils.run_utils import setup_logger_kwargs
#     logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

#     ddpg(lambda : gym.make(args.env), actor_critic=core.mlp_actor_critic,
#          ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
#          gamma=args.gamma, seed=args.seed, epochs=args.epochs,
#          logger_kwargs=logger_kwargs, lam_a=args.lam_a, lam_s=args.lam_s, eps_s=args.eps_s)
