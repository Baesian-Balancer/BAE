#!/usr/bin/env python3
# noinspection PyUnresolvedReferences

import gym
from mpi4py import MPI
from rl_smoothness.algs.baseline_ppo import logger
from rl_smoothness.algs.baseline_ppo.ppo.mlp_policy import MlpPolicy
from rl_smoothness.algs.baseline_ppo.common import set_global_seeds

def train(env_fn=None, 
          seed =0, 
          flight_log_dir=None, 
          ckpt_dir=None, 
          render=False, 
          restore_dir=None,
          ckpt_freq=1000, 
          optim_stepsize=3e-4, 
          schedule="linear", 
          gamma=0.99, 
          optim_epochs=10, 
          optim_batchsize=21, 
          horizon=2048,
          lambda_=0.95,
          entcoeff=0,
          clip_param=0.2,
          lam_a = -0.1,
          lam_s = -0.5,
          eps_s = 0.2,
          test = False,
          logger_kwargs=None,
          epochs = 40000
          ):

    from rl_smoothness.algs.baseline_ppo.ppo import pposgd_simple
    import rl_smoothness.algs.baseline_ppo.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    # if rank == 0:
    #     logger.configure()
    # else:
    #     logger.configure(format_strs=[])
    # logger.set_level(logger.DISABLED)
    workerseed = seed + 1000000 * rank
    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            #hid_size=256, num_hid_layers=3)
            #hid_size=128, num_hid_layers=2)
            hid_size=64, num_hid_layers=2)
            #hid_size=256, num_hid_layers=2)
    flight_log = None
    env = env_fn()
    if render:
        env.render()
    env.seed(workerseed)
    set_global_seeds(workerseed)
    if test:
      pposgd_simple.test(lambda: env, policy_fn, test)
    else:
      pposgd_simple.learn(lambda: env, policy_fn,
            timesteps_per_actorbatch=horizon,
            clip_param=clip_param, entcoeff=entcoeff,
            optim_epochs=optim_epochs, optim_stepsize=optim_stepsize, optim_batchsize=optim_batchsize,
            gamma=gamma, lam=lambda_, schedule=schedule,
            flight_log = flight_log,
            ckpt_dir = ckpt_dir,
            restore_dir = restore_dir,
            save_timestep_period= ckpt_freq,
            lam_a = lam_a,
            lam_s = lam_s,
            eps_s = eps_s,
            logger_kwargs = logger_kwargs,
            epochs = epochs
            )
