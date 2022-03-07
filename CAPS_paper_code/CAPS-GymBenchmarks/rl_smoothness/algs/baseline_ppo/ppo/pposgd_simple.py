from rl_smoothness.algs.baseline_ppo.common import Dataset, explained_variance, fmt_row, zipsame
# from rl_smoothness.algs.baseline_ppo import logger_kwargsr
import rl_smoothness.algs.baseline_ppo.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from rl_smoothness.algs.baseline_ppo.common.mpi_adam import MpiAdam
from rl_smoothness.algs.baseline_ppo.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
import os.path
from rl_smoothness.utils.logx import EpochLogger

def test(env_fn, policy_fn, restore_dir, tests=10):
    env = env_fn()
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space) # Construct network for new policy
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    # Restore model
    saver.restore(sess, restore_dir)
    done = False
    rewards = 0
    ob = env.reset()
    for i in range(tests):
        while not done:
            action, vpred = pi.act(False, ob)
            ob, reward, done, info = env.step(action)
            env.render()
            rewards += reward
            if done:
                ob = env.reset()
                done = False


def traj_segment_generator(pi, env, horizon, stochastic, flight_log=None, logger=None):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob = env.reset()

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...
    ep_number = 0

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    obs_nxt = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "ob_nxt": obs_nxt, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []

        ob2, rew, new, info = env.step(ac)
        i = t % horizon
        obs[i] = ob
        obs_nxt[i] = ob2
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac
        if flight_log:
            flight_log.add(cur_ep_len, ob, rew, ac, info)
        rews[i] = rew

        ob = ob2.copy()

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            logger.store(EpRet=cur_ep_ret, EpLen=cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            if flight_log:
                flight_log.save(ep_number)
                flight_log.clear()
            ep_number += 1
            ob = env.reset()
        t += 1

def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def learn(env_fn, policy_fn, *,
        timesteps_per_actorbatch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        flight_log = None,
        restore_dir = None,
        ckpt_dir = None,
        save_timestep_period = 1000,
        lam_a = -2,
        lam_s = -10,
        eps_s = 5,
        logger_kwargs=None,
        epochs):
    env = env_fn()
    max_timesteps = epochs * 5000

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_fn("oldpi", ob_space, ac_space) # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    ob_nxt = U.get_placeholder_cached(name="ob_nxt")
    ob_bar = U.get_placeholder_cached(name="ob_bar")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    if lam_a > 0:
        pol_surr += lam_a * tf.nn.l2_loss(pi.mu_nxt - pi.mu) / tf.cast(tf.shape(pi.mu)[0], float)
        # pol_surr *= 1/(0.00001 + tf.nn.l2_loss(pi.mu_nxt - pi.mu) / tf.cast(tf.shape(pi.mu)[0], float))
    if lam_s > 0:
        # pol_surr += lam_s * tf.math.sqrt(tf.nn.l2_loss(pi.mu_bar - pi.mu)) / tf.cast(tf.shape(pi.mu)[0], float)
        pol_surr += lam_s * tf.nn.l2_loss(pi.mu_bar - pi.mu) / tf.cast(tf.shape(pi.mu)[0], float)

    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ob_nxt, ob_bar, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ob_nxt, ob_bar, ac, atarg, ret, lrmult], losses)

    logger.setup_tf_saver(tf.get_default_session(), inputs={'x': ob}, outputs={'mu': pi.mu, 'v': pi.vpred})


    U.initialize()
    adam.sync()

    if restore_dir:
        ckpt = tf.train.get_checkpoint_state(restore_dir)
        if ckpt:
            # If there is one that already exists then restore it
            print("Restoring model from ", ckpt.model_checkpoint_path)
            saver.restore(tf.get_default_session(), ckpt.model_checkpoint_path)
        else:
            print("Trying to restore model from ", restore_dir, " but doesn't exist")

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch,
                                     stochastic=True, flight_log=flight_log, logger=logger)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    epochs_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"
    next_ckpt_timestep = save_timestep_period
    while True:
        if callback: callback(locals(), globals())

        end = False
        if max_timesteps and timesteps_so_far >= max_timesteps:
            end = True
        elif max_episodes and episodes_so_far >= max_episodes:
            end = True
        elif max_iters and iters_so_far >= max_iters:
            end = True
        elif max_seconds and time.time() - tstart >= max_seconds:
            end = True


        # How often should we create checkpoints
        # Because of the iterations deployed in batches this might not happen exactly
        if ((timesteps_so_far >= next_ckpt_timestep) or end):
            print("SAVEEEEEEEEEEEEE")
            print("SAVEEEEEEEEEEEEE")
            print("SAVEEEEEEEEEEEEE")
            print("SAVEEEEEEEEEEEEE")
            print("SAVEEEEEEEEEEEEE")
            print("SAVEEEEEEEEEEEEE")
            print("SAVEEEEEEEEEEEEE")
            print("SAVEEEEEEEEEEEEE")

            logger.save_state({'env': env})
            # task_name = "model.ckpt-{}".format(timesteps_so_far)
            # fname = os.path.join(ckpt_dir, task_name)
            # os.makedirs(os.path.dirname(fname), exist_ok=True)
            # saver.save(tf.get_default_session(), fname)
            next_ckpt_timestep += save_timestep_period

        if end:
            break



        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        if iters_so_far % 100 == 0:
            print("********** Iteration %i ************"%iters_so_far)

        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ob_nxt, ac, atarg, tdlamret = seg["ob"], seg["ob_nxt"], seg["ac"], seg["adv"], seg["tdlamret"]
        seg["ob_bar"] = np.random.normal(seg["ob"], eps_s)
        ob_bar = seg["ob_bar"]

        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        d = Dataset(dict(ob=ob, ob_nxt=ob_nxt, ob_bar=ob_bar, ac=ac, atarg=atarg, vtarg=tdlamret), deterministic=pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        assign_old_eq_new() # set old parameter values to new parameter values
        print("Optimizing...")
        # print(fmt_row(13, loss_names))
        # Here we do a bunch of optimization epochs over the data
        for _ in range(optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                *newlosses, g = lossandgrad(batch["ob"], batch["ob_nxt"], batch["ob_bar"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                adam.update(g, optim_stepsize * cur_lrmult)
                losses.append(newlosses)
            print(fmt_row(13, np.mean(losses, axis=0)))

        print("Evaluating losses...")
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            newlosses = compute_losses(batch["ob"], batch["ob_nxt"], batch["ob_bar"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
            losses.append(newlosses)
        meanlosses,_,_ = mpi_moments(losses, axis=0)
        # print(fmt_row(13, meanlosses))
        # for (lossval, name) in zipsame(meanlosses, loss_names):
        #     logger.log_tabular("loss_"+name, lossval)
        # logger.log_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        timesteps_so_far += sum(lens)
        episodes_so_far += len(lens)
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        if (timesteps_so_far >= epochs_so_far * 5000 ):
            epochs_so_far += 1
            logger.log_tabular('Epoch', epochs_so_far)
            # logger.log_tabular('EpRet', with_min_and_max=True)
            # logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', timesteps_so_far)
            # logger.dump_tabular()
            logger.log_tabular("AverageEpLen", np.mean(lenbuffer))
            logger.log_tabular("AverageEpRet", np.mean(rewbuffer))
            logger.log_tabular("EpThisIter", len(lens))
            iters_so_far += 1
            logger.log_tabular("EpisodesSoFar", episodes_so_far)
            logger.log_tabular("TimestepsSoFar", timesteps_so_far)
            logger.log_tabular("TimeElapsed", time.time() - tstart)
            # if MPI.COMM_WORLD.Get_rank()==0:
            logger.dump_tabular()

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
