import numpy as np
import matplotlib.pyplot as plt

def action_distribution(env, action_fn, num_tests=100, **kwargs):
    error_list = []
    outputs = []
    f,(ax1,ax2) = plt.subplots(1,2,figsize=(15,7))
    for _ in range(num_tests):
        ob = env.reset()
        done = False

        while not done:
            a = action_fn(ob, **kwargs)
            outputs.append(a[0])
            ob, _, done, _ = env.step(a)
            error_list.append(ob[0])

    outputs = np.array(outputs)
    error_list = np.array(error_list)

    ax1.hist(outputs, bins=100)
    ax1.set_xlabel('Action')
    ax1.set_ylabel('|Action|')

    H, yedges, xedges = np.histogram2d(outputs,error_list, bins=(100, 200))
    ax2.imshow(H, interpolation='nearest', origin='low',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto')
    ax2.set_xlabel('State')
    ax2.set_ylabel('Action')

    f.suptitle('Agent state-action distribution')

    return f

def test_filtered_vs_not(env, action_fn, episode=None, filter_scale=0.2, num_tests=4, **kwargs):
    f,ax = plt.subplots(4,num_tests,sharex=True,squeeze=True,figsize=(15,14))
    for test in range(num_tests):
        ob = env.reset()
        done = False

        feedback_list = []
        time_list = []
        outputs = []
        setpoint_list = []

        rew = 0

        while not done:
            a = action_fn(ob, **kwargs)
            outputs.append(a)
            ob, rew_, done, info = env.step(a)
            goal = info['goal']
            rew += rew_

            feedback_list.append(goal - ob[0])
            setpoint_list.append(goal)
            time_list.append(info['sim_time'])

        ax[0][test].set_title('Total Reward: ' + str(rew))
        ax[0][test].plot(time_list, feedback_list, label='Internal State')
        ax[0][test].plot(time_list, setpoint_list, label='Set-point')
        if test == 0:
            ax[0][test].set_ylabel('State')
        ax[0][test].grid(True)
        ax[0][test].legend()

        ax[1][test].plot(time_list, outputs, label='Actions taken')
        ax[1][test].set_xlabel('time (s)')
        if test == 0:
            ax[1][test].set_ylabel('Action')
        ax[1][test].grid(True)
        ax[1][test].legend()

    for test in range(num_tests):
        ob = env.reset()
        last_a = 0.
        done = False

        feedback_list = []
        time_list = []
        outputs = []
        setpoint_list = []

        rew = 0

        while not done:
            a = action_fn(ob, **kwargs)
            a = a*filter_scale + (1-filter_scale)*last_a
            last_a = a
            outputs.append(a)
            ob, rew_, done, info = env.step(a)
            goal = info['goal']
            rew += rew_

            feedback_list.append(goal - ob[0])
            setpoint_list.append(goal)
            time_list.append(info['sim_time'])

        ax[2][test].set_title('Total Reward: ' + str(rew))
        ax[2][test].plot(time_list, feedback_list, label='Internal State')
        ax[2][test].plot(time_list, setpoint_list, label='Set-point')
        if test == 0:
            ax[2][test].set_ylabel('Filtered State')
        ax[2][test].grid(True)
        ax[2][test].legend()

        ax[3][test].plot(time_list, outputs, label='Actions taken')
        ax[3][test].set_xlabel('time (s)')
        if test == 0:
            ax[3][test].set_ylabel('Filtered Action')
        ax[3][test].grid(True)
        ax[3][test].legend()
        
    f.suptitle('Agent tests (Unfiltered vs Filtered) - Episode '+ str(episode))
    return f

def test_agent(env, action_fn, episode=None, num_tests=4, **kwargs):
    env.seed(34343)
    f,ax = plt.subplots(2,num_tests,sharex=True,squeeze=True,figsize=(15,7))
    for test in range(num_tests):
        ob = env.reset()
        
        done = False

        feedback_list = []
        time_list = []
        outputs = []
        setpoint_list = []

        rew = 0

        while not done:
            a = action_fn(ob, **kwargs)
            outputs.append(a)
            ob, rew_, done, info = env.step(a)
            goal = info['goal']
            rew += rew_

            feedback_list.append(goal - ob[0])
            setpoint_list.append(goal)
            time_list.append(info['sim_time'])

        ax[0][test].set_title('Total Reward: ' + str(rew))
        ax[0][test].plot(time_list, feedback_list, label='Internal State')
        ax[0][test].plot(time_list, setpoint_list, label='Set-point')
        if test == 0:
            ax[0][test].set_ylabel('State')
        ax[0][test].grid(True)
        ax[0][test].legend()

        ax[1][test].plot(time_list, outputs, label='Actions taken')
        ax[1][test].set_xlabel('time (s)')
        if test == 0:
            ax[1][test].set_ylabel('Action')
        ax[1][test].grid(True)
        ax[1][test].legend()
        
    f.suptitle('Agent tests - Episode '+ str(episode))
    
    return f


def test_save(env, action_fn, alg_name, **kwargs):
    env.seed(34343)
    ob = env.reset()
    
    done = False

    feedback_list = []
    time_list = []
    outputs = []
    setpoint_list = []
    error_list = []

    rew = 0

    while not done:
        a = action_fn(ob, **kwargs)
        outputs.append(a)
        ob, rew_, done, info = env.step(a)
        goal = info['goal']
        rew += rew_

        feedback_list.append(goal - ob[0])
        error_list.append(ob[0])
        setpoint_list.append(goal)
        time_list.append(info['sim_time'])

    save_this = (feedback_list, setpoint_list, error_list, time_list, outputs, alg_name)
    import pickle
    pickle.dump( save_this, open(alg_name + ".p", "wb" ) )

