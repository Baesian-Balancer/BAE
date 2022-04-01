import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.distributions as ptd
from torch.distributions.normal import Normal
import torch.nn.functional as functional
import distributions
from distributions import mlp

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs = None):
        raise NotImplementedError

    def _get_action(self, obs = None, deterministic=False):
        pi = self._distribution(obs)
        return pi.get_actions(deterministic=deterministic)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)    # Last axis sum needed for Torch Normal distribution

    def forward(self, obs, act=None, std_mu=-1.):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)

        mu_bar = mu = self._get_action(deterministic=True)

        if std_mu > 0:
            obs_sample = torch.normal(obs, std_mu)
            mu_bar = self._get_action(obs_sample, deterministic=True)

        # Mu delta
        mu_next = torch.nn.functional.pad(mu, (0, 0, 1, 0))

        return pi, logp_a, mu, mu_next, mu_bar

class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.pi = distributions.DiagGaussianDistribution(act_dim)
        self.mu_net, self.log_std = self.pi.proba_distribution_net(obs_dim, act_dim, hidden_sizes, activation, output_activation = activation)
        self.distribution = None

    def _distribution(self, obs = None):
        if obs is not None or self.distribution is None:
            mu = self.mu_net(obs)
            self.distribution = self.pi.proba_distribution(mu, self.log_std)
        return self.distribution

class MLPGaussianSquashedActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.pi = distributions.SquashedDiagGaussianDistribution(act_dim)
        self.mu_net, self.log_std = self.pi.proba_distribution_net(obs_dim, act_dim, hidden_sizes, activation, output_activation = activation)
        self.distribution = None

    def _distribution(self, obs = None):
        if obs is not None or self.distribution is None:
            mu = self.mu_net(obs)
            self.distribution = self.pi.proba_distribution(mu, self.log_std)
        return self.distribution

class MLPBetaActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.act_dim = act_dim
        self.pi = distributions.BetaDistribution(act_dim)
        # self.alpha_net, self.beta_net = self.pi.proba_distribution_net(obs_dim, act_dim, hidden_sizes, activation)
        self.alpheta_net = self.pi.proba_distribution_net(obs_dim, act_dim, hidden_sizes, activation)
        self.distribution = None

    def _distribution(self, obs = None):
        if obs is not None or self.distribution is None:
            # alpha, beta = self.alpha_net(obs), self.beta_net(obs)
            alpheta = self.alpheta_net(obs)
            alpha, beta = alpheta.split(self.act_dim, dim=-1)
            # print(alpha, beta)
            self.distribution = self.pi.proba_distribution(alpha, beta)
        return self.distribution

class MLPBetaReparamActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.act_dim = act_dim
        self.pi = distributions.BetaDistributionReparam(act_dim)
        self.mu_net, self.kappa_net = self.pi.proba_distribution_net(obs_dim, act_dim, hidden_sizes, activation)
        self.distribution = None

    def _distribution(self, obs = None):
        if obs is not None or self.distribution is None:
            # alpha, beta = self.alpha_net(obs), self.beta_net(obs)
            mu = self.mu_net(obs)
            k = self.kappa_net(obs)
            # k = self.kappa_net([*obs, *mu])
            # print(alpha, beta)
            self.distribution = self.pi.proba_distribution(mu, k)
        return self.distribution

class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64,64), activation=nn.Tanh, dist='gaussian'):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if dist == 'gaussian':
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif dist == 'squashed_gaussian':
            self.pi = MLPGaussianSquashedActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif dist == 'beta':
            self.pi = MLPBetaActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif dist == 'reparam_beta':
            self.pi = MLPBetaActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        else:
            print(f'Distribution \'{dist}\' not supported. Defaulting to Gaussian.')
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)


        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs, deterministic=False):
        with torch.no_grad():
            if deterministic:
                a = self.pi._get_action(obs, deterministic=True)
                a = torch.clamp(a, min=-1, max=1)
                return a.numpy()
            else:
                pi = self.pi._distribution(obs)

                # Get actions from distribution (saves compute)
                a = self.pi._get_action(deterministic=False)
                a = torch.clamp(a, min=-1, max=1)

                # Get log prob with distribution
                logp_a = self.pi._log_prob_from_distribution(pi, a)

                # Value function
                v = self.v(obs)
                return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]
