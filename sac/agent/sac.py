from logging.config import dictConfig
import math
import copy
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import typing
from typing import Dict, Optional, Any
from omegaconf import OmegaConf
from agent import Agent
from agent.actor import DiagGaussianActor
from agent.critic import DoubleQCritic
from agent.rnd import RandomNetworkDistillation
import wandb
class SACAgent(Agent, nn.Module):
    """SAC algorithm."""
    def __init__(self, obs_dim, action_dim, action_range, device, critic_cfg: Dict[str, Any],
                 actor_cfg, discount, init_temperature, alpha_lr, alpha_betas,
                 actor_lr, actor_betas, actor_update_frequency, critic_lr,
                 critic_betas, critic_tau, critic_target_update_frequency,
                 batch_size, learnable_temperature):
        super().__init__()

        self.action_range = action_range
        self.device = torch.device(str(device))
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature

        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(
            self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)

        self.rnd = RandomNetworkDistillation(obs_dim, action_dim)
        
        # TODO: ADD TO CONFIG IF WORKS
        self.rnd_update_frequency = 5
        def get_train_params(model):
            trainable_parameters = []
            for name, p in model.named_parameters():
                if "pred_trunk" in name:
                    trainable_parameters.append(p)
            return trainable_parameters
        
        rnd_training_params = get_train_params(self.rnd)


        # # DoubleQCritic takes parameters: obs_dim, action_dim, hidden_dim,
        # # hidden_depth
        # # hidden_dim = critic_cfg['hidden_dim']
        # # hidden_depth = critic_cfg['hidden_depth']
        
        # # self.critic = DoubleQCritic(obs_dim, action_dim, hidden_dim, hidden_depth).to(self.device)
        # # self.critic_target = DoubleQCritic(obs_dim, action_dim, hidden_dim, hidden_depth).to(self.device)
        # # self.critic_target.load_state_dict(self.critic.state_dict())

        # # DiagGaussianActor takes parameters: obs_dim, action_dim, hidden_dim,
        # # hidden_depth, log_std_bounds
        # # self.actor = DiagGaussianActor(obs_dim, action_dim, hidden_dim,
        # #                 hidden_depth, actor_cfg["params"]['log_std_bounds']).to(self.device)

        # # self.critic = critic_cfg["critic"]
        # # self.actor = critic_cfg["actor"]
        # # self.critic_target = copy.deepcopy(critic_cfg["critic"])
        # # self.critic_target.load_state_dict(self.critic.state_dict())



        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=actor_betas)

        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=critic_betas)

        self.rnd_optimizer = torch.optim.AdamW(rnd_training_params,
                                                 lr=critic_lr,
                                                 betas=critic_betas)

        self.log_alpha_optimizer = torch.optim.AdamW([self.log_alpha],
                                                    lr=alpha_lr,
                                                    betas=alpha_betas)

        # max_lr = 0.01
        # self.actor_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.actor_optimizer, max_lr, total_steps=6000000)

        # self.critic_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.critic_optimizer, max_lr, total_steps=6000000)

        # self.log_alpha_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.log_alpha_optimizer, max_lr, total_steps=6000000)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @torch.jit.export
    def act(self, obs, sample=False):
        # obs = torch.FloatTensor(obs).to(self.device)
        obs = torch.tensor(obs).float().to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, not_done):
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1,
                             target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        if self.learnable_temperature:
            rnd_loss = self.rnd(obs, action)
            curiosity = self.rnd.normalize(rnd_loss)
            
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy - curiosity).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update_rnd(self, obs, action):
        self.rnd_optimizer.zero_grad()
        rnd_loss = self.rnd(obs, action)
        self.rnd.update_stats(rnd_loss.item())
        rnd_loss.backward()
        self.rnd_optimizer.step()
        return rnd_loss # for logging

    def update(self, replay_buffer, step):
        obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
            self.batch_size)

        self.update_critic(obs, action, reward, next_obs, not_done_no_max)

        if step % self.rnd_update_frequency == 0:
            rnd_loss = self.update_rnd(obs, action, step)
            wandb.log({"rnd_loss": rnd_loss}, step=step)
        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs)
            

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)

        # self.actor_scheduler.step()
        # self.critic_scheduler.step()
        # self.log_alpha_scheduler.step()
