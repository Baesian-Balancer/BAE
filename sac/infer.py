#!/usr/bin/env python3
import wandb
import gym
import numpy as np
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
from plotting import PlotUtils
import hydra

class Workspace(object):
    def __init__(self, cfg,agent_cfg):
        
        if not os.path.isfile(cfg.cp_path):
            raise FileExistsError("Model checkpoint path does not exist")

        self.cfg = cfg

        dir = os.getcwd()
        self.plotting = PlotUtils("tester", dir)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = utils.make_env(cfg)
        self.env.render('human')
        print(self.env.observation_space.shape[0])
        print(self.env.action_space.shape[0])
        print(self.env.action_space.low.min())
        print(self.env.action_space.high.max()),

        agent_cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        agent_cfg.agent.params.action_dim = self.env.action_space.shape[0]
        agent_cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        # print(agent_cfg.agent)
        self.agent = hydra.utils.instantiate(agent_cfg.agent)
        print(self.agent)
        checkpoint = torch.load(cfg.cp_path)
        
        self.agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.agent.actor.load_state_dict(checkpoint['actor_state_dict'])

        replay_buffer_capacity = self.cfg.num_eval_episodes * 10_000
        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          replay_buffer_capacity,
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
            # while not done:
            for i in range(700):
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                next_obs, reward, done, _ = self.env.step(action)
                print(action, _)
                self.plotting.add_action(action)

                done = float(done)
                done_no_max = 0 if step + 1 == 1000 else done
                episode_reward += reward
                self.replay_buffer.add(obs, action, reward, next_obs, done,
                        done_no_max)
                
                obs = next_obs
                # print(obs)
                episode_reward += reward
                step+=1
            average_episode_reward += episode_reward
        average_episode_reward /= self.cfg.num_eval_episodes
        print(average_episode_reward)
        self.plotting.plot_temporal_action_change()
        self.plotting.plot_action_histogram()    

        # Save all the data
        filename = 'replaybuffer_data.obj'
        with open(filename, 'wb') as fp:
            pkl.dump(self.replay_buffer, fp)

    def run(self):
        self.evaluate()

    def train(self):
        for i in range(self.cfg.num_train_updates):
            self.agent.update(self.replay_buffer, 0)
        self.save()

    def save(self):
        PATH = os.path.splitext(self.cfg.cp_path)[0] + "_updated.pt"
        torch.save({
        'actor_state_dict': self.agent.actor.state_dict(),
        'critic_state_dict': self.agent.critic.state_dict(),
        'rnd_state_dict': self.agent.critic.state_dict()
        }, PATH)

def main(cfg):

    if cfg.wandb_on:
        wandb.init(project=cfg.wandb_project,entity=cfg.wandb_user)

    agent_cfg = OmegaConf.load('config/agent/sac.yaml')
    agent_cfg.agent.params.device = cfg.device
    agent_cfg.agent.params.batch_size = 0
    workspace = Workspace(cfg,agent_cfg)
    # workspace.run()
    workspace.train()
    workspace.save()


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
        '--cp_path',
        default = 'exp/best_model.pt',
        type = str    
    )

    parser.add_argument(
        '--num_eval_episodes',
        default= 10,
        type = int  
    )

    parser.add_argument(
        '--num_train_updates',
        default= 10,
        type = int
    )
    args = parser.parse_args()

    main(args)
