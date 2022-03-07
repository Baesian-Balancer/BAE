import numpy as np
import tensorflow as tf
import gym
import time
import rl_smoothness.algs.cores.ddpg_core as core
from rl_smoothness.algs.cores.ddpg_core import get_vars

import pprint as pp

import os

from rl_smoothness.utils.get_env import GetEnv
from rl_smoothness.utils.testing_utils import action_distribution, test_filtered_vs_not, test_agent, test_save

import os.path
import datetime
import subprocess

from signal import signal, SIGINT
from sys import exit

import matplotlib.pyplot as plt

from rl_smoothness.train_alg import TrainAlg

# ===========================
#   Tensorflow Summary Ops
# ===========================

def build_summaries():
    episode_reward = tf.Variable(0.)
    q_max = tf.Variable(0.)

    tf.summary.scalar("EpisodeReward", episode_reward)
    tf.summary.scalar("q_max", q_max)

    summary_vars = [episode_reward, q_max]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

"""

Deep Deterministic Policy Gradient (DDPG)

"""
def ddpg(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0, 
         eps_per_epoch=10, epochs=100, replay_size=int(1e6), gamma=0.99, 
         polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=1000, 
         act_noise=0.1, logger_kwargs=dict(), save_freq=1, lam_a=40, lam_s=200,
         args={'training_dir':'./tmp_train', 'ckpt_dir':'./tmp_train/checkpoints'}):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Deterministically computes actions
                                           | from policy given states.
            ``q``        (batch,)          | Gives the current estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q_pi``     (batch,)          | Gives the composition of ``q`` and 
                                           | ``pi`` for states in ``x_ph``: 
                                           | q(x, pi(x)).
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to DDPG.

        seed (int): Seed for random number generators.

        eps_per_epoch (int): Number of episodes of interaction
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        act_noise (float): Stddev for Gaussian exploration noise added to 
            policy at training time. (At test time, no noise is added.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        args (dict): Additional information used to facilitate logging.

    """

    tf.set_random_seed(seed)
    np.random.seed(seed)

    env_dt = 0.02
    env_max_time = 3.0
    max_ep_len = int(env_max_time / env_dt)
    
    ad_test_period = args['action_distr_test_period']
    if ad_test_period < 0: do_ad_test = False
    else: do_ad_test = True

    test_period = args['test_period']
    compare_filtered = args['compare_filtered']

    ad_test_period = args['action_distr_test_period']
    if ad_test_period < 0: do_ad_test = False
    else: do_ad_test = True

    if args['test_target']: test_target = True
    else: test_target=False

    if args['act_mode'] == 'absolute':
        act_mode = 'A'
    else:
        act_mode = 'R'

    env = GetEnv(env_type=args['env_type'], env_mode=args['env_mode'], dt=env_dt, max_time=env_max_time, action_type=act_mode, args=args)
    if args['test_env_type'] == args['env_type']: test_env = env
    else: 
        test_env = GetEnv(env_type=args['test_env_type'], env_mode=args['env_mode'], dt=env_dt, max_time=env_max_time, action_type=act_mode, args=args)

    env.seed(int(args['seed']))

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    x_ph = tf.placeholder(dtype=tf.float32, name='x', shape=(None,obs_dim) if obs_dim else (None,))
    x2_ph = tf.placeholder(dtype=tf.float32, name='x_t', shape=(None,obs_dim) if obs_dim else (None,))
    xnxt_ph = tf.placeholder(dtype=tf.float32, name='x_nxt', shape=(None,obs_dim) if obs_dim else (None,))
    xbar_ph = tf.placeholder(dtype=tf.float32, name='x_bar', shape=(None,obs_dim) if obs_dim else (None,))
    a_ph, xnxt2_dummy_ph, xbar2_dummy_ph, r_ph, d_ph = core.placeholders(act_dim, obs_dim, obs_dim, None, None)

    # Main outputs from computation graph
    with tf.variable_scope('main'):
        pi, pi_nxt, pi_bar, q, q_pi = actor_critic(x_ph, a_ph, xnxt_ph, xbar_ph, **ac_kwargs)
    
    # Target networks
    with tf.variable_scope('target'):
        # Note that the action placeholder going to actor_critic here is 
        # irrelevant, because we only need q_targ(s, pi_targ(s)).
        pi_targ, _, _, _, q_pi_targ  = actor_critic(x2_ph, a_ph, xnxt2_dummy_ph, xbar2_dummy_ph, **ac_kwargs)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['main/pi', 'main/q', 'main'])
    print('\nNumber of parameters: \t pi: %d, \t q: %d, \t total: %d\n'%var_counts)

    # Bellman backup for Q function
    backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*q_pi_targ)

    # DDPG losses
    pi_loss = -tf.reduce_mean(q_pi)
    if args['reg_a']:
        pi_loss += lam_a*tf.nn.l2_loss(pi_nxt - pi)/tf.cast(tf.shape(pi)[0], float)
    if args['reg_s']:
        pi_loss += lam_s*tf.nn.l2_loss(pi_bar - pi)/tf.cast(tf.shape(pi)[0], float)
    q_loss = tf.reduce_mean((q-backup)**2)

    # Separate train ops for pi, q
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr)
    q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lr)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))
    train_q_op = q_optimizer.minimize(q_loss, var_list=get_vars('main/q'))

    # Polyak averaging for target variables
    target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    sess = tf.Session()

    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()
    writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)

    # Setup model saving
    saver = tf.train.Saver(max_to_keep=epochs)

    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    def get_action(o, noise_scale=0., target=False, **kwargs):
        if target:
            a = sess.run(pi_targ, feed_dict={x2_ph: o.reshape(1,-1)})[0]
        else:
            a = sess.run(pi, feed_dict={x_ph: o.reshape(1,-1)})[0]
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def get_q(o,a):
        return sess.run(q, feed_dict={x_ph: o.reshape(1,-1), a_ph: a.reshape(1,-1)})

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    
    total_eps = eps_per_epoch * epochs

    print("Done with setup. Entering main loop")
    ep_counter = 0
    t = 0
    last_save = None
    q_max = None
    # Main loop: collect experience in env and update/log each epoch
    while t < max_ep_len*total_eps:
        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy (with some noise, via act_noise). 
        """
        if t > start_steps:
            a = get_action(o, act_noise)
            q_out = get_q(o,a)
            q_max = max(q_out, q_max) if q_max else q_out
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1


        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        t+=1

        if d or (ep_len == max_ep_len):
            """
            Perform all DDPG updates at the end of the trajectory,
            in accordance with tuning done by TD3 paper authors.
            """

            ep_counter+= 1

            for _ in range(max_ep_len):
                batch = replay_buffer.sample_batch(batch_size)
                feed_dict = {x_ph: batch['obs1'],
                             x2_ph: batch['obs2'],
                             xnxt_ph: batch['obs2'],
                             xnxt2_dummy_ph: batch['obs2'],
                             a_ph: batch['acts'],
                             r_ph: batch['rews'],
                             d_ph: batch['done']
                            }
                
                if args['reg_s']:
                    feed_dict[xbar_ph] = np.random.normal(feed_dict[x_ph], args['s_eps'])

                # Q-learning update
                outs = sess.run([q_loss, q, train_q_op], feed_dict)

                # Policy update
                outs = sess.run([pi_loss, train_pi_op, target_update], feed_dict)

            summary_feed = {summary_vars[0]: ep_ret[0], summary_vars[1]: q_max[0] if q_max else None }
            summary_str = sess.run(summary_ops, feed_dict=summary_feed)
            writer.add_summary(summary_str, ep_counter)
            writer.flush()

            print('Episode: '+ str(ep_counter) + '; Reward: ' + str(ep_ret))

            if do_ad_test and (ep_counter%ad_test_period == 0):
                fig1_1 = action_distribution(env, get_action, target=test_target)
                if (args['env_type'] != 'degenerate') and (args['env_type'] != args['test_env_type']):
                    fig1_2 = action_distribution(test_env, get_action, target=test_target)
            
            if ep_counter%test_period == 0:
                fig2_1 = test_agent(env, get_action, ep_counter, target=test_target)
                test_save(env, get_action, "DDPG", target=test_target)
                if (args['env_type'] != 'degenerate') and (args['env_type'] != args['test_env_type']):
                    fig2_2 = test_agent(test_env, get_action, ep_counter, target=test_target)

            if compare_filtered and (ep_counter%100 == 0):
                fig3_1 = test_filtered_vs_not(env, get_action, ep_counter, filter_scale=0.33, target=test_target)
                if (args['env_type'] != 'degenerate') and (args['env_type'] != args['test_env_type']):
                    fig3_1 = test_filtered_vs_not(test_env, get_action, ep_counter, filter_scale=0.33, target=test_target)

            if not args['save']:
                plt.show()

            q_max = None
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # End of epoch wrap-up
        if t > 0 and ep_counter > 0 and ep_counter % eps_per_epoch == 0 and last_save != ep_counter:
            epoch = t // eps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs-1):
                task_name = "model.ckpt-{}".format(t)
                model_save_path = os.path.join(args['ckpt_dir'],task_name)
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                save_path = saver.save(sess, model_save_path)
                print("Checkpoint saved in path: %s" % save_path)
            last_save = ep_counter

from rl_smoothness.train_alg import TrainAlg

class DDPGAlg(TrainAlg):
    
    def name(self):
        return "ddpg"

    def add_args(self, parser):
        from rl_smoothness.utils.common_parser import add_common_args
        add_common_args(parser)
        return parser

    def train(self, args):
        ddpg(lambda : gym.make(args.env), actor_critic=core.mlp_actor_critic, \
             #ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
             gamma=args['gamma'], seed=args['seed'], epochs=args['epochs'], args=args)