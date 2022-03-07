import gym
import numpy as np
import rl_smoothness.utils.fourier as fourier
import math
from gym.envs.registration import registry, register
import matplotlib.pyplot as plt

register(
    id='Pendulum-v1',
    entry_point='rl_smoothness.envs.Pendulum:PendulumEnv',
    max_episode_steps=200,
)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v1')
    parser.add_argument('--tests', type=int, default=3)
    args = parser.parse_args()
    env = gym.make(args.env)

    ob_space = env.observation_space
    ac_space = env.action_space
    done = False
    rewards = 0
    ob = env.reset()
    # env.env.state = np.array([math.pi, 0])
    N = 200
    freqs = np.linspace(0., .5, N // 2)
    print(freqs)
    amplitude = 1
    freq_response = np.zeros(len(freqs))
    for i in range(len(freqs)):
        count = 0
        done = False
        obs = []
        for j in range(len(freqs)*2):
            # print(env.state)
            # action, vpred = pi.act(False, ob)
            action = amplitude*math.cos(freqs[i]*count*2*math.pi)
            # print(action)
            count += 1
            ob, reward, done, info = env.step(np.array([action]))
            obs.append(ob[2])
            # env.render()
            rewards += reward
            # if done:
        ob = env.reset()
        print("reset"+str(i))
        # env.env.state = np.array([math.pi, 0])

        frs, amplitudes = fourier.fourier_transform(obs, T=1)
        print(amplitudes.shape)
        print(freq_response.shape)
        freq_response[i] = amplitudes[i]


    fourier.plot(freqs, freq_response)
    plt.show()