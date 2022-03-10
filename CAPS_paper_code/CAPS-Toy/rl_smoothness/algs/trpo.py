import numpy as np
import tensorflow as tf
import gym
import time
import rl_smoothness.algs.cores.trpo_core as core
from rl_smoothness.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from rl_smoothness.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

import os
import pprint as pp

from rl_smoothness.utils.get_env import GetEnv
from rl_smoothness.utils.testing_utils import action_distribution, test_filtered_vs_not, test_agent, test_save

import os.path
import datetime
import subprocess

import matplotlib.pyplot as plt

EPS = 1e-8

def build_summaries():
    episode_reward = tf.Variable(0.)

    tf.summary.scalar("EpisodeReward", episode_reward)

    summary_vars = [episode_reward]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

class GAEBuffer:
    """
    A buffer for storing trajectories experienced by a TRPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, info_shapes, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.nxt_obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.info_bufs = {k: np.zeros([size] + list(v), dtype=np.float32) for k,v in info_shapes.items()}
        self.sorted_info_keys = core.keys_as_sorted_list(self.info_bufs)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, nxt_obs, rew, val, logp, info):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.nxt_obs_buf[self.ptr] = nxt_obs
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        for i, k in enumerate(self.sorted_info_keys):
            self.info_bufs[k][self.ptr] = info[i]
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
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.nxt_obs_buf, self.adv_buf, self.ret_buf, 
                self.logp_buf] + core.values_as_sorted_list(self.info_bufs)



def trpo(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0, 
         eps_per_epoch=10, epochs=100, gamma=0.99, delta=0.01, vf_lr=1e-3,
         train_v_iters=80, damping_coeff=0.1, cg_iters=10, backtrack_iters=10, 
         backtrack_coeff=0.8, lam=0.97, max_ep_len=1000, logger_kwargs=dict(), lam_a=1, lam_s=5,
         save_freq=10, algo='trpo', args={'training_dir':'./tmp_train', 'ckpt_dir':'./tmp_train/checkpoints'}):
    """
    Trust Region Policy Optimization 

    (with support for Natural Policy Gradient)

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:

            ============  ================  ========================================
            Symbol        Shape             Description
            ============  ================  ========================================
            ``pi``        (batch, act_dim)  | Samples actions from policy given 
                                            | states.
            ``logp``      (batch,)          | Gives log probability, according to
                                            | the policy, of taking actions ``a_ph``
                                            | in states ``x_ph``.
            ``logp_pi``   (batch,)          | Gives log probability, according to
                                            | the policy, of the action sampled by
                                            | ``pi``.
            ``info``      N/A               | A dict of any intermediate quantities
                                            | (from calculating the policy or log 
                                            | probabilities) which are needed for
                                            | analytically computing KL divergence.
                                            | (eg sufficient statistics of the
                                            | distributions)
            ``info_phs``  N/A               | A dict of placeholders for old values
                                            | of the entries in ``info``.
            ``d_kl``      ()                | A symbol for computing the mean KL
                                            | divergence between the current policy
                                            | (``pi``) and the old policy (as 
                                            | specified by the inputs to 
                                            | ``info_phs``) over the batch of 
                                            | states given in ``x_ph``.
            ``v``         (batch,)          | Gives the value estimate for states
                                            | in ``x_ph``. (Critical: make sure 
                                            | to flatten this!)
            ============  ================  ========================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to TRPO.

        seed (int): Seed for random number generators.

        eps_per_epoch (int): Number of episodes of interaction for the agent and
            the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        delta (float): KL-divergence limit for TRPO / NPG update. 
            (Should be small for stability. Values like 0.01, 0.05.)

        vf_lr (float): Learning rate for value function optimizer.

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        damping_coeff (float): Artifact for numerical stability, should be 
            smallish. Adjusts Hessian-vector product calculation:
            
            .. math:: Hv \\rightarrow (\\alpha I + H)v

            where :math:`\\alpha` is the damping coefficient. 
            Probably don't play with this hyperparameter.

        cg_iters (int): Number of iterations of conjugate gradient to perform. 
            Increasing this will lead to a more accurate approximation
            to :math:`H^{-1} g`, and possibly slightly-improved performance,
            but at the cost of slowing things down. 

            Also probably don't play with this hyperparameter.

        backtrack_iters (int): Maximum number of steps allowed in the 
            backtracking line search. Since the line search usually doesn't 
            backtrack, and usually only steps back once when it does, this
            hyperparameter doesn't often matter.

        backtrack_coeff (float): How far back to step during backtracking line
            search. (Always between 0 and 1, usually above 0.5.)

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        algo: Either 'trpo' or 'npg': this code supports both, since they are 
            almost the same.

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
    
    if args['act_mode'] == 'absolute':
        act_mode = 'A'
    else:
        act_mode = 'R'

    env = GetEnv(env_type=args['env_type'], env_mode=args['env_mode'], dt=env_dt, max_time=env_max_time, action_type=act_mode, args=args)
    if args['test_env_type'] == args['env_type']: test_env = env
    else: 
        test_env = GetEnv(env_type=args['test_env_type'], env_mode=args['env_mode'], dt=env_dt, max_time=env_max_time, action_type=act_mode, args=args)
    
    env.seed(int(args['seed']))

    steps_per_epoch = max_ep_len * eps_per_epoch

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    
    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    x_ph = tf.placeholder(dtype=tf.float32, name='x', shape=(None,obs_dim[0]) if obs_dim else (None,))
    xnxt_ph = tf.placeholder(dtype=tf.float32, name='x_nxt', shape=(None,obs_dim[0]) if obs_dim else (None,))
    xbar_ph = tf.placeholder(dtype=tf.float32, name='x_bar', shape=(None,obs_dim[0]) if obs_dim else (None,))
    a_ph = core.placeholder_from_space(env.action_space)
    adv_ph, ret_ph, logp_old_ph = core.placeholders(None, None, None)

    # Main outputs from computation graph, plus placeholders for old pdist (for KL)
    mu, pi, mu_nxt, pi_nxt, mu_bar, pi_bar, logp, logp_pi, info, info_phs, d_kl, v = actor_critic(x_ph, a_ph, xnxt_ph, xbar_ph, **ac_kwargs)

    # Need all placeholders in *this* order later (to zip with data from buffer)
    all_phs = [x_ph, a_ph, xnxt_ph, adv_ph, ret_ph, logp_old_ph] + core.values_as_sorted_list(info_phs)

    # Every step, get: action, value, logprob, & info for pdist (for computing kl div)
    get_action_ops = [pi, v, logp_pi] + core.values_as_sorted_list(info)

    # Experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    info_shapes = {k: v.shape.as_list()[1:] for k,v in info_phs.items()}
    buf = GAEBuffer(obs_dim, act_dim, local_steps_per_epoch, info_shapes, gamma, lam)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['pi', 'v'])
    # logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # TRPO losses
    ratio = tf.exp(logp - logp_old_ph)          # pi(a|s) / pi_old(a|s)
    pi_loss = -tf.reduce_mean(ratio * adv_ph)
    if args['reg_a']:
        pi_loss += lam_a*tf.nn.l2_loss(mu_nxt - mu)/tf.cast(tf.shape(mu)[0], float)
    if args['reg_s']:
        pi_loss += lam_s*tf.nn.l2_loss(mu_bar - mu)/tf.cast(tf.shape(mu)[0], float)
    v_loss = tf.reduce_mean((ret_ph - v)**2)

    # Optimizer for value function
    train_vf = MpiAdamOptimizer(learning_rate=vf_lr).minimize(v_loss)

    # Symbols needed for CG solver
    pi_params = core.get_vars('pi')
    gradient = core.flat_grad(pi_loss, pi_params)
    v_ph, hvp = core.hessian_vector_product(d_kl, pi_params)
    if damping_coeff > 0:
        hvp += damping_coeff * v_ph

    # Symbols for getting and setting params
    get_pi_params = core.flat_concat(pi_params)
    set_pi_params = core.assign_params_from_flat(v_ph, pi_params)

    sess = tf.Session()

    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()
    writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)

    sess.run(tf.global_variables_initializer())

    # Sync params across processes
    sess.run(sync_all_params())

    # Setup model saving
    saver = tf.train.Saver(max_to_keep=epochs)
    # logger.setup_tf_saver(sess, inputs={'x': x_ph}, outputs={'pi': pi, 'v': v})

    def cg(Ax, b):
        """
        Conjugate gradient algorithm
        (see https://en.wikipedia.org/wiki/Conjugate_gradient_method)
        """
        x = np.zeros_like(b)
        r = b.copy() # Note: should be 'b - Ax(x)', but for x=0, Ax(x)=0. Change if doing warm start.
        p = r.copy()
        r_dot_old = np.dot(r,r)
        for _ in range(cg_iters):
            z = Ax(p)
            alpha = r_dot_old / (np.dot(p, z) + EPS)
            x += alpha * p
            r -= alpha * z
            r_dot_new = np.dot(r,r)
            p = r + (r_dot_new / r_dot_old) * p
            r_dot_old = r_dot_new
        return x

    def update():
        # Prepare hessian func, gradient eval
        inputs = {k:v for k,v in zip(all_phs, buf.get())}
        if args['reg_s']:
            inputs[xbar_ph] = np.random.normal(inputs[x_ph], args['s_eps'])
        Hx = lambda x : mpi_avg(sess.run(hvp, feed_dict={**inputs, v_ph: x}))
        g, pi_l_old, v_l_old = sess.run([gradient, pi_loss, v_loss], feed_dict=inputs)
        g, pi_l_old = mpi_avg(g), mpi_avg(pi_l_old)

        # Core calculations for TRPO or NPG
        x = cg(Hx, g)
        alpha = np.sqrt(2*delta/(np.dot(x, Hx(x))+EPS))
        old_params = sess.run(get_pi_params)

        def set_and_eval(step):
            sess.run(set_pi_params, feed_dict={v_ph: old_params - alpha * x * step})
            return mpi_avg(sess.run([d_kl, pi_loss], feed_dict=inputs))

        if algo=='npg':
            # npg has no backtracking or hard kl constraint enforcement
            kl, pi_l_new = set_and_eval(step=1.)

        elif algo=='trpo':
            # trpo augments npg with backtracking line search, hard kl
            for j in range(backtrack_iters):
                kl, pi_l_new = set_and_eval(step=backtrack_coeff**j)
                if kl <= delta and pi_l_new <= pi_l_old:
                    # logger.log('Accepting new params at step %d of line search.'%j)
                    # logger.store(BacktrackIters=j)
                    break

                if j==backtrack_iters-1:
                    # logger.log('Line search failed! Keeping old params.')
                    # logger.store(BacktrackIters=j)
                    kl, pi_l_new = set_and_eval(step=0.)

        # Value function updates
        for _ in range(train_v_iters):
            sess.run(train_vf, feed_dict=inputs)
        v_l_new = sess.run(v_loss, feed_dict=inputs)

        # Log changes from update
        # logger.store(LossPi=pi_l_old, LossV=v_l_old, KL=kl,
        #              DeltaLossPi=(pi_l_new - pi_l_old),
        #              DeltaLossV=(v_l_new - v_l_old))

    def get_action(ob, deterministic=False, **kwargs):
        if deterministic:
            a = sess.run(mu, feed_dict={x_ph: ob.reshape(1,-1)})[0]
        else:
            agent_outs = sess.run(get_action_ops, feed_dict={x_ph: ob.reshape(1,-1)})
            a = agent_outs[0][0]
        return a

    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    ep_counter = 0
    last_save = None

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            agent_outs = sess.run(get_action_ops, feed_dict={x_ph: o.reshape(1,-1)})
            a, v_t, logp_t, info_t = agent_outs[0][0], agent_outs[1], agent_outs[2], agent_outs[3:]

            o2, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(o, a, o2, r, v_t, logp_t, info_t)
            # logger.store(VVals=v_t)

            # Update obs (critical!)
            o = o2

            terminal = d or (ep_len == max_ep_len)
            if terminal or (t==local_steps_per_epoch-1):
                ep_counter += 1

                if not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len)
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = 0 if d else sess.run(v, feed_dict={x_ph: o.reshape(1,-1)})
                buf.finish_path(last_val)

                summary_feed = {summary_vars[0]: ep_ret[0]}
                summary_str = sess.run(summary_ops, feed_dict=summary_feed)
                writer.add_summary(summary_str, ep_counter)
                writer.flush()

                # if terminal:
                #     # only save EpRet / EpLen if trajectory finished
                #     logger.store(EpRet=ep_ret, EpLen=ep_len)

                print('Episode: '+ str(ep_counter) + '; Reward: ' + str(ep_ret))

                if do_ad_test and (ep_counter%ad_test_period == 0):
                    fig1_1 = action_distribution(env, get_action, deterministic=deterministic)
                    if (args['env_type'] != 'degenerate') and (args['env_type'] != args['test_env_type']):
                        fig1_2 = action_distribution(test_env, get_action, deterministic=deterministic)

                
                if ep_counter%test_period == 0:
                    fig2_1 = test_agent(env, get_action, ep_counter, deterministic=deterministic)
                    test_save(env, get_action, "TRPO", deterministic=deterministic)
                    if (args['env_type'] != 'degenerate') and (args['env_type'] != args['test_env_type']):
                        fig2_2 = test_agent(test_env, get_action, ep_counter, deterministic=deterministic)

                if compare_filtered and (ep_counter%100 == 0):
                    fig3_1 = test_filtered_vs_not(env, get_action, ep_counter, filter_scale=0.33, deterministic=deterministic)
                    if (args['env_type'] != 'degenerate') and (args['env_type'] != args['test_env_type']):
                        fig3_2 = test_filtered_vs_not(test_env, get_action, ep_counter, filter_scale=0.33, deterministic=deterministic)

                if not args['save']:
                    plt.show()

                o, ep_ret, ep_len = env.reset(), 0, 0

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            # logger.save_state({'env': env}, None)
            task_name = "model.ckpt-{}".format(t)
            model_save_path = os.path.join(args['ckpt_dir'],task_name)
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            save_path = saver.save(sess, model_save_path)
            print("Checkpoint saved in path: %s" % save_path)
            # logger.save_state({'env': env}, None)
        last_save = ep_counter

        # Perform TRPO or NPG update!
        update()

from rl_smoothness.train_alg import TrainAlg

class TRPOAlg(TrainAlg):
    
    def name(self):
        return "trpo"

    def add_args(self, parser):
        from rl_smoothness.utils.common_parser import add_common_args
        add_common_args(parser)
        return parser

    def train(self, args):
        mpi_fork(1)  # run parallel code with mpi (but don't actually because it breaks things?)
        trpo(lambda : gym.make(args['env']), actor_critic=core.mlp_actor_critic, \
             # sac_kwargs=dict(hidden_sizes=[args['hid']]*args['l']), 
             gamma=args['gamma'], seed=args['seed'], epochs=args['epochs'], args=args)
