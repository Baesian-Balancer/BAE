import numpy as np
import tensorflow as tf
import gym
import time

import rl_smoothness.algs.cores.td3_core as core
from rl_smoothness.algs.cores.td3_core import get_vars

import os
import pprint as pp

from rl_smoothness.utils.get_env import GetEnv
from rl_smoothness.utils.testing_utils import action_distribution, test_filtered_vs_not, test_agent, test_save

import os.path

import matplotlib.pyplot as plt

# ===========================
#   Tensorflow Summary Ops
# ===========================

def build_summaries():
    episode_reward = tf.Variable(0.)

    tf.summary.scalar("EpisodeReward", episode_reward)

    summary_vars = [episode_reward]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
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
TD3 (Twin Delayed DDPG)
"""
def td3(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0, 
        eps_per_epoch=10, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, pi_lr=1e-4, q_lr=1e-4, batch_size=64, start_steps=1000, 
        act_noise=0.1, target_noise=0.2, noise_clip=0.5, policy_delay=2, lam_a=2, lam_s=10,
        logger_kwargs=dict(), save_freq=1, args={'training_dir':'./tmp_train', 'ckpt_dir':'./tmp_train/checkpoints'}):
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
            ``q1``       (batch,)          | Gives one estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q2``       (batch,)          | Gives another estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q1_pi``    (batch,)          | Gives the composition of ``q1`` and 
                                           | ``pi`` for states in ``x_ph``: 
                                           | q1(x, pi(x)).
            ===========  ================  ======================================
        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to TD3.
        seed (int): Seed for random number generators.
        eps_per_epoch (int): Number of episodes of interaction for the agent and
            the environment in each epoch.
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
        target_noise (float): Stddev for smoothing noise added to target 
            policy.
        noise_clip (float): Limit for absolute value of target policy 
            smoothing noise.
        policy_delay (int): Policy will only be updated once every 
            policy_delay times for each update of the Q-networks.
        max_ep_len (int): Maximum length of trajectory / episode / rollout.
        logger_kwargs (dict): Keyword args for EpochLogger.
        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
        args (dict): Additional information used to facilitate logging.
    """

    # logger = EpochLogger(**logger_kwargs)
    # logger.save_config(locals())

    tf.set_random_seed(seed)
    np.random.seed(seed)

    # env, test_env = env_fn(), env_fn()
    env_dt = 0.02
    env_max_time = 3.0
    max_ep_len = env_max_time / env_dt

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
    # test_env.seed(int(args['seed']))

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # print('Obs dim: '+ str(obs_dim))
    # print('Act dim: '+ str(act_dim))

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

    # Target policy network
    with tf.variable_scope('target'):
        pi_targ, _, _, _, _, _  = actor_critic(x2_ph, a_ph, xnxt2_dummy_ph, xbar2_dummy_ph, **ac_kwargs)

    target_variables = []
    for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        if v.name.find('pi') > -1:
            target_variables.append(v)

    # Main outputs from computation graph
    with tf.variable_scope('main'):
        pi, pi_nxt, pi_bar, q1, q2, q1_pi = actor_critic(x_ph, a_ph, xnxt_ph, xbar_ph, **ac_kwargs)
    
    # Target Q networks
    with tf.variable_scope('target', reuse=True):

        # Target policy smoothing, by adding clipped noise to target actions
        epsilon = tf.random_normal(tf.shape(pi_targ), stddev=target_noise)
        epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
        a2 = pi_targ + epsilon
        a2 = tf.clip_by_value(a2, -act_limit, act_limit)

        # Target Q-values, using action from target policy
        _, _, _, q1_targ, q2_targ, _ = actor_critic(x2_ph, a2, xnxt2_dummy_ph, xbar2_dummy_ph, **ac_kwargs)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['main/pi', 'main/q1', 'main/q2', 'main'])
    print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d, \t total: %d\n'%var_counts)

    # Bellman backup for Q functions, using Clipped Double-Q targets
    min_q_targ = tf.minimum(q1_targ, q2_targ)
    backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*min_q_targ)

    # TD3 losses
    pi_loss = -tf.reduce_mean(q1_pi)
    if args['reg_a']:
        pi_loss += lam_a*tf.nn.l2_loss(pi_nxt - pi)/tf.cast(tf.shape(pi)[0], float)
    if args['reg_s']:
        pi_loss += lam_s*tf.nn.l2_loss(pi_bar - pi)/tf.cast(tf.shape(pi)[0], float)
    q1_loss = tf.reduce_mean((q1-backup)**2)
    q2_loss = tf.reduce_mean((q2-backup)**2)
    q_loss = q1_loss + q2_loss

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
        # o = np.array([o])
        if target:
            a = sess.run(pi_targ, feed_dict={x2_ph: o.reshape(1,-1)})[0]
        else:
            a = sess.run(pi, feed_dict={x_ph: o.reshape(1,-1)})[0]
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_eps = eps_per_epoch * epochs

    print("Done with setup. Entering main loop")
    ep_counter = 0
    t = 0
    last_save = None
    # Main loop: collect experience in env and update/log each epoch
    while t < max_ep_len*total_eps:
        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy (with some noise, via act_noise). 
        """
        if t > start_steps:
            a = get_action(o, act_noise)
        else:
            a = env.action_space.sample()


        # Step the env
        o2, r, d, s_info = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        # d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        t+=1

        if d or (ep_len == max_ep_len):
            """
            Perform all TD3 updates at the end of the trajectory
            (in accordance with source code of TD3 published by
            original authors).
            """
            ep_counter+= 1

            for j in range(ep_len):
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
                
                q_step_ops = [q_loss, q1, q2, train_q_op]
                outs = sess.run(q_step_ops, feed_dict)
                # logger.store(LossQ=outs[0], Q1Vals=outs[1], Q2Vals=outs[2])

                if j % policy_delay == 0:
                    # Delayed policy update
                    outs = sess.run([pi_loss, train_pi_op, target_update], feed_dict)
                    # print(outs)
                    # logger.store(LossPi=outs[0])

            summary_feed = {summary_vars[0]: ep_ret[0]}
            summary_str = sess.run(summary_ops, feed_dict=summary_feed)
            writer.add_summary(summary_str, ep_counter)
            writer.flush()
            # print(get_vars('train'))
                # print(var)

            print('Episode: '+ str(ep_counter) + '; Reward: ' + str(ep_ret))

            if do_ad_test and (ep_counter%ad_test_period == 0):
                fig1_1 = action_distribution(env, get_action, target=test_target)
                if (args['env_type'] != 'degenerate') and (args['env_type'] != args['test_env_type']):
                    fig1_2 = action_distribution(test_env, get_action, target=test_target)
            
            if ep_counter%test_period == 0:
                fig2_1 = test_agent(env, get_action, ep_counter, target=test_target)
                test_save(env, get_action, "TD3", target=test_target)
                if (args['env_type'] != 'degenerate') and (args['env_type'] != args['test_env_type']):
                    fig2_2 = test_agent(test_env, get_action, ep_counter, target=test_target)

            if compare_filtered and (ep_counter%100 == 0):
                fig3_1 = test_filtered_vs_not(env, get_action, ep_counter, filter_scale=0.33, target=test_target)
                if (args['env_type'] != 'degenerate') and (args['env_type'] != args['test_env_type']):
                    fig3_2 = test_filtered_vs_not(test_env, get_action, ep_counter, filter_scale=0.33, target=test_target)

            if not args['save']:
                plt.show()

            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # End of epoch wrap-up
        if t > 0 and ep_counter > 0 and ep_counter % eps_per_epoch == 0 and last_save != ep_counter:
            epoch = ep_counter // eps_per_epoch
            # print(t,ep_counter,eps_per_epoch,epoch)
            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs-1):
                task_name = "model.ckpt-{}".format(t)
                model_save_path = os.path.join(args['ckpt_dir'],task_name)
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                save_path = saver.save(sess, model_save_path)
                print("Checkpoint saved in path: %s" % save_path)
                # logger.save_state({'env': env}, None)
            last_save = ep_counter

from rl_smoothness.train_alg import TrainAlg

class TD3Alg(TrainAlg):
    
    def name(self):
        return "td3"

    def add_args(self, parser):
        from rl_smoothness.utils.common_parser import add_common_args
        add_common_args(parser)
        return parser

    def train(self, args):
        td3(lambda : gym.make(args['env']), actor_critic=core.mlp_actor_critic, \
            #ac_kwargs=dict(hidden_sizes=[args['hid']]*args['l']),
            gamma=args['gamma'], seed=args['seed'], epochs=args['epochs'], args=args)