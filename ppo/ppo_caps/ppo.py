from ast import arg
import wandb

import numpy as np
import torch
from tqdm.auto import tqdm
from torch.optim import Adam
import gym
import gym_os2r
from gym_os2r import randomizers
import time
import core
import functools

import datetime
import os

def make_env(env_id):

    def make_env_from_id(env_id: str, **kwargs) -> gym.Env:
        return gym.make(env_id, **kwargs)

    # Create a partial function passing the environment id
    create_env = functools.partial(make_env_from_id, env_id=env_id)
    env = randomizers.monopod_no_rand.MonopodEnvNoRandomizer(env=create_env)
    # env = randomizers.monopod.MonopodEnvRandomizer(env=create_env)

    # Enable the rendering
    # env.render('human')

    # Initialize the seed
    print(env)
    return env

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs_next_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.mu_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.mu_bar_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.mu_delta_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, obs_next, act, rew, val, logp, mu, mu_bar):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.obs_next_buf[self.ptr] = obs_next
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.mu_buf[self.ptr] = mu
        self.mu_bar_buf[self.ptr] = mu_bar
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        # Find the difference in expected policy action
        mu = self.mu_buf[path_slice] # A
        mu_next = np.vstack([np.zeros(mu.shape[1]), mu])
        self.mu_delta_buf[path_slice] = mu_next[:-1] - mu

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        # adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        # self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, obs_next=self.obs_next_buf, act=self.act_buf,
                    ret=self.ret_buf, adv=self.adv_buf, logp=self.logp_buf,
                    mu=self.mu_buf, mu_delta=self.mu_delta_buf, mu_bar=self.mu_bar_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

def evaluate(o, ac,env,config, max_steps_per_ep, cur_step):
    # Main loop: collect experience in env and update/log each epoch
    progress_bar = tqdm(range(config["eval_epochs"]),desc='Evaluation Epochs')
    eval_ret = 0
    eval_ret_norm = 0
    eval_len = 0
    ep_ret = 0
    ep_len = 0

    for epoch in range(config["eval_epochs"]):
        for t in range(max_steps_per_ep):
            with torch.no_grad():
                a = ac.step(torch.as_tensor(o, dtype=torch.float32),eval=True)
            o, r, d, _ = env.step(a)

            ep_ret += r
            ep_len += 1

            timeout = ep_len == max_steps_per_ep
            terminal = d or timeout

            if terminal:
                eval_ret_norm += ep_ret/max_steps_per_ep
                eval_ret += ep_ret
                eval_len += 1
                o, ep_ret, ep_len = env.reset(), 0, 0
        progress_bar.update(1)
    eval_ret /= eval_len
    eval_ret_norm /= eval_len

    wandb.log({"evaluation normalized reward":eval_ret_norm, "evaluation reward":eval_ret}, step=cur_step)
    return o, eval_ret,eval_len

def ppo(env_fn, config ,actor_critic=core.MLPActorCritic, ac_kwargs=dict()):

    if not os.path.isdir(config["save_dir"]):
        os.makedirs(config["save_dir"])

    # Random seed
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    if config['load_model_path'] is not None:
        checkpoint = torch.load(config['load_model_path'])
        ac.load_state_dict(checkpoint['actor_state_dict'])

    # Set up experience buffer
    local_steps_per_epoch = int(config["steps_per_epoch"])
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, config["gamma"], config["lam"])

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, obs_next, act, adv, logp_old, mu, mu_bar, mu_delta = data['obs'], data['obs_next'], data['act'], data['adv'], data['logp'], data['mu'], data['mu_bar'], data['mu_delta']

        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        ratio_adv = ratio * adv
        clip_adv = torch.clamp(ratio, 1-config["clip_ratio"], 1+config["clip_ratio"]) * adv
        loss_pi = -(torch.min(ratio_adv, clip_adv)).mean()

        temporal_smoothness, spatial_smoothness = 0, 0
        if config['lam_a'] > 0:
            temporal_smoothness = torch.norm(mu_delta)
            loss_pi += config['lam_a'] * temporal_smoothness
        if config['lam_s'] > 0:
            spatial_smoothness = torch.norm(mu - mu_bar)
            loss_pi += config['lam_s'] * spatial_smoothness
        if config['lam_o'] > 0:
            state_smoothness = torch.norm(obs - obs_next)
            loss_pi += config['lam_o'] * state_smoothness

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+config["clip_ratio"]) | ratio.lt(1-config["clip_ratio"])
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac, ts=temporal_smoothness)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=config["pi_lr"])
    vf_optimizer = Adam(ac.v.parameters(), lr=config["vf_lr"])


    def update(step):
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        wandb.log(pi_info_old, step=step)

        # Train policy with multiple steps of gradient descent
        for i in range(config["train_pi_iters"]):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            loss_pi.backward()
            pi_optimizer.step()

        # Value function learning
        for i in range(config["train_v_iters"]):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            vf_optimizer.step()

    # Prepare for interaction with environment
    o, ep_ret, ep_len = env.reset(), 0, 0

    current_max_ep_len = config["start_ep_len"]
    update_ep_len = (config["max_ep_len"]-config["start_ep_len"]) // config["epochs"]

    bst_eval_ret = 0;
    # Main loop: collect experience in env and update/log each epoch
    progress_bar = tqdm(range(config["epochs"]),desc='Training Epoch')
    for epoch in range(config["epochs"]):
        for t in range(local_steps_per_epoch):
            a, v, logp, mu, mu_bar = ac.step(torch.as_tensor(o, dtype=torch.float32), std_mu=config['eps_s'])
            next_o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(o, next_o, a, r, v, logp, mu, mu_bar)

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == current_max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                else:
                    wandb.log({"training episode normalized reward":ep_ret/current_max_ep_len, "training episode reward":ep_ret}, step=epoch*local_steps_per_epoch + t)

                if timeout or epoch_ended:
                    _, v, _, _, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(v)

                o, ep_ret, ep_len = env.reset(), 0, 0
        progress_bar.update(1)
        # Save model
        if epoch %config["save_freq"] == 0:
            PATH = config["save_dir"] + "checkpoint_model_step_" + str(epoch*local_steps_per_epoch + t) + ".pt"
            torch.save({
                'actor_state_dict': ac.state_dict(),
            }, PATH)

        # Perform PPO update!
        update(step=epoch*local_steps_per_epoch + t)
        o, eval_ret,_ = evaluate(o, ac,env,config, current_max_ep_len, (epoch + 1)*local_steps_per_epoch)

        if bst_eval_ret < eval_ret:
            bst_eval_ret = eval_ret
            PATH = config["save_dir"] + "best_model_step_" + str(epoch*local_steps_per_epoch + t) + ".pt"
            torch.save({
                'actor_state_dict': ac.state_dict(),
            }, PATH)

        current_max_ep_len += update_ep_len
        current_max_ep_len = min(current_max_ep_len, config["max_ep_len"])
        wandb.log({"max ep length": current_max_ep_len}, step=epoch*local_steps_per_epoch + t + 1)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Monopod-balance-v4')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.97)
    parser.add_argument('--clip_ratio', type=float, default=0.2)
    parser.add_argument('--target_kl', type=float, default=0.01)
    parser.add_argument('--pi_lr', type=float, default=3e-4)
    parser.add_argument('--vf_lr',type=float, default=1e-3)
    parser.add_argument('--train_pi_iters', type=int, default=80)
    parser.add_argument('--train_v_iters', type=int, default=80)
    parser.add_argument('--seed', '-s', type=int, default=42)
    parser.add_argument('--steps_per_epoch', type=int, default=20000)
    parser.add_argument('--max_ep_len', type=int, default=6000)
    parser.add_argument('--start_ep_len', type=int, default=3000)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--eval_epochs', type=int, default=3)
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--exp_name', type=str, default='ppo caps')
    parser.add_argument('--save_dir', type=str, default=f'exp/{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}/')
    parser.add_argument('--load_model_path', type=str, default=None)

    parser.add_argument('--lam_a', type=float, help='Regularization coeffecient on action smoothness (valid > 0)', default=1)
    parser.add_argument('--lam_s', type=float, help='Regularization coeffecient on state mapping smoothness (valid > 0)', default=1)
    parser.add_argument('--eps_s', type=float, help='Variance coeffecient on state mapping smoothness (valid > 0)', default=0.05)
    parser.add_argument('--lam_o', type=float, help='Regularization coeffecient on observation state mapping smoothness (valid > 0)', default=1)

    args = parser.parse_args()
    wandb.init(project="openSim2Real", entity="dawon-horvath", config=args)

    config = wandb.config

    ppo(lambda : make_env(config["env"]), config, actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[config["hid"]]*config["l"]),)
