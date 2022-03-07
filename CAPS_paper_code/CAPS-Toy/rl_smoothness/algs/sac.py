import numpy as np
import tensorflow as tf
import gym
import time
import rl_smoothness.algs.cores.sac_core as core
from rl_smoothness.algs.cores.sac_core import get_vars

import os
import pprint as pp

from rl_smoothness.utils.get_env import GetEnv
from rl_smoothness.utils.testing_utils import action_distribution, test_filtered_vs_not, test_agent, test_save

import os.path
import datetime

import matplotlib.pyplot as plt

def build_summaries():
    episode_reward = tf.Variable(0.)

    tf.summary.scalar("EpisodeReward", episode_reward)

    summary_vars = [episode_reward]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
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



def sac(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0, 
        eps_per_epoch=10, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1, lam_a=20, lam_s=100,
        args={'training_dir':'./tmp_train', 'ckpt_dir':'./tmp_train/checkpoints'}):
    """
    Soft Actor-Critic (SAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``mu``       (batch, act_dim)  | Computes mean actions from policy
                                           | given states.
            ``pi``       (batch, act_dim)  | Samples actions from policy given 
                                           | states.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``. Critical: must be differentiable
                                           | with respect to policy parameters all
                                           | the way through action sampling.
            ``q1``       (batch,)          | Gives one estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q2``       (batch,)          | Gives another estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to SAC.

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

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

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
    max_ep_len = env_max_time / env_dt

    deterministic = args['deterministic']
    
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
    x2_ph = tf.placeholder(dtype=tf.float32, name='x_2', shape=(None,obs_dim) if obs_dim else (None,))
    xnxt_ph = tf.placeholder(dtype=tf.float32, name='x_nxt', shape=(None,obs_dim) if obs_dim else (None,))
    xbar_ph = tf.placeholder(dtype=tf.float32, name='x_bar', shape=(None,obs_dim) if obs_dim else (None,))
    a_ph, xnxt2_dummy_ph, xbar2_dummy_ph, r_ph, d_ph = core.placeholders(act_dim, obs_dim, obs_dim, None, None)

    # Main outputs from computation graph
    with tf.variable_scope('main'):
        mu, pi, mu_nxt, pi_nxt, mu_bar, pi_bar, logp_pi, q1, q2 = actor_critic(x_ph, a_ph, xnxt_ph, xbar_ph, **ac_kwargs)

    with tf.variable_scope('main', reuse=True):
        # compose q with pi, for pi-learning
        _, _, _, _, _, _, _, q1_pi, q2_pi = actor_critic(x_ph, pi, xnxt2_dummy_ph, xbar2_dummy_ph, **ac_kwargs)

        # get actions and log probs of actions for next states, for Q-learning
        _, pi_next, _, _, _, _, logp_pi_next, _, _ = actor_critic(x2_ph, a_ph,  xnxt2_dummy_ph, xbar2_dummy_ph, **ac_kwargs)
    
    # Target value network
    with tf.variable_scope('target'):
        # target q values, using actions from *current* policy
        _, _, _, _, _, _, _, q1_targ, q2_targ  = actor_critic(x2_ph, pi_next, xnxt2_dummy_ph, xbar2_dummy_ph, **ac_kwargs)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['main/pi', 'main/q1', 'main/q2', 'main'])
    print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d, \t total: %d\n'%var_counts)

    # Min Double-Q:
    min_q_pi = tf.minimum(q1_pi, q2_pi)
    min_q_targ = tf.minimum(q1_targ, q2_targ)

    # Entropy-regularized Bellman backup for Q functions, using Clipped Double-Q targets
    q_backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*(min_q_targ - alpha * logp_pi_next))

    # Soft actor-critic losses
    pi_loss = tf.reduce_mean(alpha * logp_pi - min_q_pi)
    if args['reg_a']:
        pi_loss += lam_a*tf.nn.l2_loss(mu_nxt - mu)/tf.cast(tf.shape(mu)[0], float)
    if args['reg_s']:
        pi_loss += lam_s*tf.nn.l2_loss(mu_bar - mu)/tf.cast(tf.shape(mu)[0], float)
    q1_loss = 0.5 * tf.reduce_mean((q_backup - q1)**2)
    q2_loss = 0.5 * tf.reduce_mean((q_backup - q2)**2)
    value_loss = q1_loss + q2_loss

    # Policy train op 
    # (has to be separate from value train op, because q1_pi appears in pi_loss)
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))

    # Value train op
    # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
    value_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    value_params = get_vars('main/q')
    with tf.control_dependencies([train_pi_op]):
        train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)

    # Polyak averaging for target variables
    # (control flow because sess.run otherwise evaluates in nondeterministic order)
    with tf.control_dependencies([train_value_op]):
        target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                  for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # All ops to call during one training step
    step_ops = [pi_loss, q1_loss, q2_loss, q1, q2, logp_pi, 
                train_pi_op, train_value_op, target_update]

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


    def get_action(o, deterministic=False, **kwargs):
        act_op = mu if deterministic else pi
        return sess.run(act_op, feed_dict={x_ph: o.reshape(1,-1)})[0]

    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    total_eps = eps_per_epoch * epochs

    ep_counter = 0
    last_save = None

    max_t = int(max_ep_len*total_eps)

    # Main loop: collect experience in env and update/log each epoch
    for t in range(max_t):

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

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        # d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2
        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            ep_counter += 1
            print('Episode: '+ str(ep_counter) + '; Reward: ' + str(ep_ret))

            summary_feed = {summary_vars[0]: ep_ret[0]}
            summary_str = sess.run(summary_ops, feed_dict=summary_feed)
            writer.add_summary(summary_str, ep_counter)
            writer.flush()

            # Test the performance of the deterministic version of the agent.
            if do_ad_test and (ep_counter%ad_test_period == 0):
                fig1_1 = action_distribution(env, get_action, deterministic=deterministic)
                if (args['env_type'] != 'degenerate') and (args['env_type'] != args['test_env_type']):
                    fig1_2 = action_distribution(test_env, get_action, deterministic=deterministic)

            
            if ep_counter%test_period == 0:
                fig2_1 = test_agent(env, get_action, ep_counter, deterministic=deterministic)
                test_save(env, get_action, "SAC", deterministic=deterministic)
                if (args['env_type'] != 'degenerate') and (args['env_type'] != args['test_env_type']):
                    fig2_2 = test_agent(test_env, get_action, ep_counter, deterministic=deterministic)

            if compare_filtered and (ep_counter%100 == 0):
                fig3_1 = test_filtered_vs_not(env, get_action, ep_counter, filter_scale=0.33, deterministic=deterministic)
                if (args['env_type'] != 'degenerate') and (args['env_type'] != args['test_env_type']):
                    fig3_2 = test_filtered_vs_not(test_env, get_action, ep_counter, filter_scale=0.33, deterministic=deterministic)

            if not args['save']:
                plt.show()
            
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                feed_dict = {x_ph: batch['obs1'],
                             x2_ph: batch['obs2'],
                             xnxt_ph: batch['obs2'],
                             xnxt2_dummy_ph: batch['obs2'],
                             a_ph: batch['acts'],
                             r_ph: batch['rews'],
                             d_ph: batch['done'],
                            }
                if args['reg_s']:
                    feed_dict[xbar_ph] = np.random.normal(feed_dict[x_ph], args['s_eps'])
                
                outs = sess.run(step_ops, feed_dict)

        # End of epoch wrap-up
        if t > 0 and ep_counter > 0 and ep_counter % eps_per_epoch == 0 and last_save != ep_counter:
            epoch = ep_counter // eps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                task_name = "model.ckpt-{}".format(t)
                model_save_path = os.path.join(args['ckpt_dir'],task_name)
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                save_path = saver.save(sess, model_save_path)
                print("Checkpoint saved in path: %s" % save_path)
                # logger.save_state({'env': env}, None)
            last_save = ep_counter


from rl_smoothness.train_alg import TrainAlg

class SACAlg(TrainAlg):
    
    def name(self):
        return "sac"

    def add_args(self, parser):
        from rl_smoothness.utils.common_parser import add_common_args
        add_common_args(parser)
        return parser

    def train(self, args):
        sac(lambda : gym.make(args['env']), actor_critic=core.mlp_actor_critic, \
            #ac_kwargs=dict(hidden_sizes=[args['hid']]*args['l']),
            gamma=args['gamma'], seed=args['seed'], epochs=args['epochs'], args=args)