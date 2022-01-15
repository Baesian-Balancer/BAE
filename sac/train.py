#!/usr/bin/env python3
import wandb
import gym
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import pickle as pkl
# from BB_gym_Envs import randomizers
import functools
from replay_buffer import ReplayBuffer
import utils
from argparse import ArgumentParser
from omegaconf import OmegaConf

import hydra

class Workspace(object):
    def __init__(self, cfg,agent_cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        if not os.path.isdir(cfg.cp_dir) and cfg.save_cp:
            os.mkdir(cfg.cp_dir)

        self.cfg = cfg

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = utils.make_env(cfg)
        print(self.env.observation_space.shape[0])
        print(self.env.action_space.shape[0])


        agent_cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        agent_cfg.agent.params.action_dim = self.env.action_space.shape[0]
        agent_cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(agent_cfg.agent)

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          int(cfg.replay_buffer_capacity),
                                          self.device)
        self.step = 0

        self.best_avg_reward = 0

    def evaluate(self):
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            done = False
            episode_reward = 0
            step = 0
            progress_bar = tqdm(range(1000),desc='Evaluation')
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward
                step+=1
                progress_bar.update(1)

            average_episode_reward += episode_reward
        average_episode_reward /= self.cfg.num_eval_episodes
        
        if self.cfg.wandb_on:
            wandb.log({"average reward":average_episode_reward})
        if average_episode_reward >= self.best_avg_reward and self.cfg.save_cp:
            PATH = self.cfg.cp_dir + "best_model.pt"
            torch.save({
            'actor_state_dict': self.agent.actor.state_dict(),
            'critic_state_dict': self.agent.critic.state_dict(),
            }, PATH)
            self.best_avg_reward = average_episode_reward

    def run(self):
        episode, episode_reward, done = 0, 0, True
        progress_bar = tqdm(range(int(self.cfg.num_train_steps)),desc='Train')
        while self.step < self.cfg.num_train_steps:
            if done:
                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.evaluate()

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

            # sample action for data collection
            if self.step < self.cfg.exploration_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.exploration_steps:
                self.agent.update(self.replay_buffer, self.step)

            next_obs, reward, done, _ = self.env.step(action)
            # allow infinite bootstrap
            done = float(done)
            # done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            done_no_max = 0 if episode_step + 1 == 1000 else done
            episode_reward += reward
            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)
            
            obs = next_obs
            episode_step += 1
            self.step += 1
            progress_bar.update(1)

def main(cfg):
    if cfg.wandb_on:
        wandb.init(project=cfg.wandb_project,entity=cfg.wandb_user)
        
    agent_cfg = OmegaConf.load('config/agent/sac.yaml')
    agent_cfg.agent.params.device = cfg.device
    agent_cfg.agent.params.batch_size = cfg.batch_size
    workspace = Workspace(cfg,agent_cfg)
    workspace.run()


if __name__ == '__main__':

    parser = ArgumentParser(add_help=False)

    parser.add_argument(
        '--wandb_on',
        default = False,
        action = 'store_true'  
    )
    parser.add_argument(
        '--wandb_project',
        default = 'capstone',
        type = str    
    )
    parser.add_argument(
        '--wandb_user',
        default = 'nickioan',
        type = str    
    )

    parser.add_argument(
        '--device',
        default = 'cuda',
        type = str    
    )

    parser.add_argument(
        '--seed',
        default = 42,
        type = int    
    )

    parser.add_argument(
        '--env_id',
        default = 'Monopod-balance-v1',
        type = str   
    )

    parser.add_argument(
        '--save_cp',
        default = False,
        action = 'store_true'   
    )

    parser.add_argument(
        '--cp_dir',
        default = 'exp/',
        type = str    
    )

    parser.add_argument(
        '--log_frequency',
        default = 10,
        type = int  
    )

    parser.add_argument(
        '--eval_frequency',
        default = 5000,
        type = int  
    )

    parser.add_argument(
        '--num_eval_episodes',
        default= 10,
        type = int  
    )

    parser.add_argument(
        '--replay_buffer_capacity',
        default = 1e6,
        type = int
    )

    parser.add_argument(
        '--batch_size',
        default = 256,
        type = int
    )

    parser.add_argument(
        '--exploration_steps',
        default = 5000,
        type = int
    )

    parser.add_argument(
        '--num_train_steps',
        default = 1e6,
        type = int
    )

    args = parser.parse_args()

    main(args)
