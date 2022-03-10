import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import noise

import warnings

class DegEnv():
    def __init__(self, dt=0.02, max_time=3., goal=0.1, seed=None, **kwargs):
        self.sim_time = 0.
        self.max_time = max_time
        self.dt = dt

        self.obs_high = np.array([10., 10.])
        self.max_act = np.array([10.])

        self.observation_space = spaces.Box(low=-self.obs_high, high=self.obs_high)
        self.action_space = spaces.Box(low=-self.max_act, high=self.max_act)

        self.goal = np.array([goal])
        self.state = self.goal
        self.done = False
        self.seed(seed)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_goal(self, goal):
        self.goal = np.array([goal])

    def step(self, u):
        if self.done:
            warnings.warn('Max sim time exceeded. Time to reset')

        u = np.clip(u, -self.max_act, self.max_act)
        
        self.state += 2*np.random.rand()-1
        # self.state = np.clip(self.state + u, -self.obs_high[1], self.obs_high[1])
        # self.state = u

        self.sim_time += self.dt

        if self.sim_time > self.max_time:
            self.done = True

        reward = -np.abs(self.goal - u)

        state_return = None
        state_return = np.concatenate((self.goal-self.state, self.state))

        return state_return, reward, self.done, {'goal':self.goal, 'sim_time':self.sim_time-self.dt}

    def reset(self):
        self.sim_time = 0.
        self.state = self.goal.copy()
        self.done = False

        state_return = np.concatenate((self.goal-self.state, self.state))

        return state_return