import envs.PerlinEnv as PerlinEnv
import envs.StepEnv as StepEnv
from PIDactor import PID

import numpy as np
import matplotlib.pyplot as plt

def test_pid(P = 0.2,  I = 0.0, D= 0.0, max_sim_time=4):
    """Self-test PID class

    .. note::
        ...
        for i in range(1, END):
            pid.update(feedback)
            output = pid.output
            if pid.SetPoint > 0:
                feedback += (output - (1/i))
            if i>9:
                pid.SetPoint = 1
            time.sleep(0.02)
        ---
    """
    env = PerlinEnv.StateEnv(max_time=max_sim_time)
    state = env.reset()
    error = state[0]

    pid = PID(P, I, D)

    done = False
    feedback = 0

    feedback_list = []
    time_list = []
    outputs = []
    setpoint_list = []

    while not done:
        a = pid.act(error)
        outputs.append(a)
        state, _, done, info = env.step(a)
        error = state[0]

        goal = info['goal']

        feedback_list.append((goal-error))
        setpoint_list.append(goal)
        time_list.append(info['sim_time'])

    # time_sm = np.array(time_list)
    # time_smooth = np.linspace(time_sm.min(), time_sm.max(), 300)

    # feedback_smooth = spline(time_list, feedback_list, time_smooth)
    # Using make_interp_spline to create BSpline
    # helper_x3 = make_interp_spline(time_list, feedback_list)
    # feedback_smooth = helper_x3(time_smooth)

    f,(ax1,ax2) = plt.subplots(2,1,sharex=True, squeeze=True)
    ax1.set_title('Test PID')
    ax1.plot(time_list, feedback_list, label='Internal State')
    ax1.plot(time_list, setpoint_list, label='Set-point')
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('State')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(time_list, outputs, label='Actions taken')
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('Action')
    ax2.grid(True)
    ax2.legend()

    plt.show()

if __name__ == "__main__":
    # test_pid(1.2, 1, 0.001, L=50)
    test_pid(0.3, 0.0, 0.00, max_sim_time=10)
#    test_pid(0.8, L=50)