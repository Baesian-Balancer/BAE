from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import core as core
import datetime

from tqdm.auto import tqdm
import wandb
import gym_os2r
from gym_os2r import randomizers
import functools

def make_env(env_id):

    def make_env_from_id(env_id: str, **kwargs) -> gym.Env:
        return gym.make(env_id, **kwargs)

    # Create a partial function passing the environment id
    create_env = functools.partial(make_env_from_id, env_id=env_id)
    env = randomizers.monopod_no_rand.MonopodEnvNoRandomizer(env=create_env)

    # Enable the rendering
    # env.render('human')

    # Initialize the seed
    print(env)
    return env


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

def evaluate(o, ac,env,config, max_steps_per_ep, cur_step):
    # Main loop: collect experience in env and update/log each epoch
    progress_bar = tqdm(range(config["eval_epochs"]),desc='Evaluation Epochs')
    eval_ret = 0
    eval_ret_norm = 0
    eval_len = 0

    o, ep_ret, ep_len = env.reset(), 0, 0
    for epoch in range(config["eval_epochs"]):
        for t in range(max_steps_per_ep):

            with torch.no_grad():
                a, v, logp = ac.act(torch.as_tensor(obs, dtype=torch.float32),deterministic=True)
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

def sac(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        total_train_steps=1_000_000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000,
        update_after=1000, update_every=5, num_test_episodes=1, max_ep_len=10_000,
        test_freq=5, save_freq=1, config=None):

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    # logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data['obs']
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    # Set up model saving
    # logger.setup_pytorch_saver(ac)

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        # logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Record things
        # logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32),
                      deterministic)

    # Prepare for interaction with environment
    # total_steps = steps_per_epoch * epochs
    # print(f"total steps: {total_steps}\n")
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    episode = 0
    bst_eval_ret = 0
    # Main loop: collect experience in env and update/log each epoch
    progress_bar = tqdm(range(total_train_steps), desc='Training Steps')
    for t in range(total_train_steps):
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        if t > start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1
        progress_bar.update(1)

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            # logger.store(EpRet=ep_ret, EpLen=ep_len)
            episode += 1
                wandb.log({"training episode normalized reward":ep_ret/max_ep_len, "training episode reward":ep_ret}, step=t)
            o, ep_ret, ep_len = env.reset(), 0, 0

            o, eval_ret,_ = evaluate(o, ac, env, config, max_ep_length, t)
            if bst_eval_ret < eval_ret:
                bst_eval_ret = eval_ret
                PATH = config["save_dir"] + f'best_model_step_{t}.pt'
                torch.save({
                    'actor_state_dict': ac.state_dict(),
                }, PATH)

        if epoch %config["save_freq"] == 0:
            PATH = config["save_dir"] + "checkpoint_model_step_" + str(epoch*local_steps_per_epoch + t) + ".pt"
            torch.save({
            'actor_state_dict': ac.state_dict(),
            }, PATH)

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Monopod-balance-v1')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    parser.add_argument('--save_dir', type=str, default=f'exp/{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}/')
    parser.add_argument('--save_freq', type=str, default=5)
    args = parser.parse_args()


    args = parser.parse_args()
    wandb.init(project="openSim2Real", entity="dawon-horvath", config=args)

    config = wandb.config

    sac(lambda : make_env(config['env']), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[config['hid']]*config['l']),
        gamma=config['gamma'], seed=config['seed'], test_freq=5, epochs=config['epochs'], config=config)

    print("ALL DONE EVERYTHING")
