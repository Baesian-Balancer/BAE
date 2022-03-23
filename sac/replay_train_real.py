from argparse import ArgumentParser
from click import Argument
import numpy as np
import torch
import pickle as pkl
import gym
import hydra
import os 
from omegaconf import OmegaConf
import utils

from argparse import ArgumentParser


class Workspace:
    def __init__(self, cfg,agent_cfg):

        if not os.path.isfile(cfg.cp_path):
            raise FileExistsError("Model checkpoint path does not exist")

        self.cfg = cfg

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = gym.make(cfg.env_id)


        agent_cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        agent_cfg.agent.params.action_dim = self.env.action_space.shape[0]
        agent_cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(agent_cfg.agent)

        checkpoint = torch.load(cfg.cp_path)

        self.agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.agent.rnd.load_state_dict(checkpoint['actor_state_dict'])
        
        # Load replay buffer data
        with open('replay_buffer.obj', 'rb') as fp:
            replay_buffer = pkl.load(fp)

    def train(self):
        for i in range(self.cfg.num_train_updates):
            self.agent.update()
        self.save_model()

    def save_model(self):
        PATH = os.path.splitext(self.cfg.cp_path)[0] + "_updated.pt"
        torch.save({
        'actor_state_dict': self.agent.actor.state_dict(),
        'critic_state_dict': self.agent.critic.state_dict(),
        'rnd_state_dict': self.agent.critic.state_dict()
        }, PATH)
        pass


def main(cfg):
    agent_cfg = OmegaConf.load('config/agent/sac.yaml')
    agent_cfg.agent.params.device = cfg.device
    agent_cfg.agent.params.batch_size = 0
    workspace = Workspace(cfg,agent_cfg)
    workspace.run()

if __name__ == '__main__':

    parser = ArgumentParser(add_help=False)

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
        default = 'Monopod-stand-v1',
        type = str
    )

    parser.add_argument(
        '--cp_path',
        default = 'exp/best_model.pt',
        type = str
    )

    parser.add_argument(
        '--num_train_updates',
        default= 10,
        type = int
    )