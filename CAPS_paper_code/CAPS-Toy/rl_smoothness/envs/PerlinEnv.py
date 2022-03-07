import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import noise

import warnings

class BaseEnv():
    def __init__(self, dt=0.02, max_time=3., p_aggression=1., continuous=True, discont_prob=0.01, discont_range=1, action_type='R', seed=None, **kwargs):
        self.sim_time = 0.
        self.max_time = max_time
        self.dt = dt
        self.aggression = p_aggression
        self.continuous = continuous
        self.discont_prob = discont_prob
        self.discont_range = discont_range

        self.action_type = action_type
        
        # Dummy to be over-written
        self.obs_high = np.array([0.])
        self.seed(seed)

        self.offset = self.np_random.rand()*1e5
        self.goal = None
        self.update_goal()
        self.goal_set = False
        self.state = self.goal
        self.done = False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def update_goal(self):
        if not self.continuous:
            if self.np_random.rand() < self.discont_prob:
                self.offset += self.np_random.rand()*self.discont_range
        self.goal = np.array([noise.pnoise1(self.sim_time*self.aggression + self.offset, 4)*self.obs_high[0]])

    def step(self, u):
        raise NotImplementedError('Step function not implemented for this Perlin Env')

    def reset(self):
        self.sim_time = 0.
        self.offset = self.np_random.rand(1)[0]*1e5
        self.update_goal()
        self.goal_set = False
        self.state = self.goal.copy()
        self.done = False

class StateEnv(BaseEnv):
    def __init__(self, dt=0.02, max_time=3., p_aggression=1., continuous=True, discont_prob=0.01, discont_range=1, action_type='R', seed=None, **kwargs):
        super().__init__(dt=dt, max_time=max_time, p_aggression=p_aggression, continuous=continuous, discont_prob=discont_prob, discont_range=discont_range, action_type=action_type, seed=seed, **kwargs)
        
        if self.action_type == 'A': #absolute
            self.obs_high = np.array([10., 10.])
            self.max_act = np.array([10.])
        elif self.action_type == 'R': #relative
            self.obs_high = np.array([10.])
            self.max_act = np.array([1.])
        else:
            raise ValueError('Action type for PerlinEnv.StateEnv not recognized. Accepted types are "R" for relative actions and "A" for absolute actions')

        self.observation_space = spaces.Box(low=-self.obs_high, high=self.obs_high)
        self.action_space = spaces.Box(low=-self.max_act, high=self.max_act)
        print('Initialized with type ' + self.action_type)

    def step(self, u):
        if self.done:
            warnings.warn('Max sim time exceeded. Time to reset')

        self.update_goal()
        u = np.clip(u, -self.max_act, self.max_act)

        if self.action_type == 'A':
            self.state = u
        elif self.action_type == 'R':
            self.state += u

        self.sim_time += self.dt

        if self.sim_time > self.max_time:
            self.done = True

        # reward = -np.abs(0.13 - u) # For action-based reward test
        reward = -np.abs(self.goal - self.state)

        state_return = None
        if self.action_type == 'A':
            state_return = np.concatenate((self.goal-self.state, self.state))
        elif self.action_type == 'R':
            state_return = self.goal-self.state

        return state_return, reward, self.done, {'goal':self.goal, 'sim_time':self.sim_time-self.dt}

    def reset(self):
        super().reset()

        state_return = None
        if self.action_type == 'A':
            state_return = np.concatenate((self.goal-self.state, self.state))
        elif self.action_type == 'R':
            state_return = self.goal-self.state

        return state_return
        
class VelocityEnv(BaseEnv):
    def __init__(self, dt=0.02, max_time=3., p_aggression=1., continuous=True, discont_prob=0.01, discont_range=1, action_type='R', seed=None, **kwargs):
        super().__init__(dt=dt, max_time=max_time, p_aggression=p_aggression, continuous=continuous, discont_prob=discont_prob, discont_range=discont_range, action_type=action_type, seed=seed, **kwargs)
        
        if self.action_type == 'A': #absolute
            self.obs_high = np.array([10.])
            # self.obs_high = np.array([10., 10., 50.])
            self.max_act = np.array([50.])
        elif self.action_type == 'R': #relative
            self.obs_high = np.array([10., 50.])
            self.max_act = np.array([50.])
        else:
            raise ValueError('Action type for PerlinEnv.VelocityEnv not recognized. Accepted types are "R" for relative actions and "A" for absolute actions')

        self.observation_space = spaces.Box(low=-self.obs_high, high=self.obs_high)
        self.action_space = spaces.Box(low=-self.max_act, high=self.max_act)

        self.velocity = np.array([0.])

        print('Initialized with type ' + self.action_type)

    def step(self, u):
        if self.done:
            warnings.warn('Max sim time exceeded. Time to reset')
        
        self.update_goal()
        u = np.clip(u, -self.max_act, self.max_act)

        if self.action_type == 'A':
            self.velocity = u
        elif self.action_type == 'R':
            self.velocity += u

        self.state += self.velocity * self.dt
        self.sim_time += self.dt

        if self.sim_time > self.max_time:
            self.done = True
        
        reward = -np.abs(self.goal - self.state)

        state_return = None
        if self.action_type == 'A':
            state_return = self.goal-self.state
            # state_return = np.concatenate((self.goal-self.state, self.state, self.velocity))
        elif self.action_type == 'R':
            state_return = np.concatenate([self.goal-self.state, self.velocity])

        return state_return, reward, self.done, {'goal':self.goal, 'sim_time':self.sim_time-self.dt}

    def reset(self):
        super().reset()

        self.velocity = np.array([0.])
        
        state_return = None
        if self.action_type == 'A':
            state_return = self.goal-self.state
            # state_return = np.concatenate((self.goal-self.state, self.state, self.velocity))
        elif self.action_type == 'R':
            state_return = np.concatenate((self.goal-self.state, self.velocity))

        return state_return

class AccelerationEnv(BaseEnv):
    def __init__(self, dt=0.02, max_time=3., p_aggression=1., continuous=True, discont_prob=0.01, discont_range=1, action_type='R', seed=None, **kwargs):
        super().__init__(dt=dt, max_time=max_time, p_aggression=p_aggression, continuous=continuous, discont_prob=discont_prob, discont_range=discont_range, action_type=action_type, seed=seed, **kwargs)
        
        if self.action_type == 'A': #absolute
            self.obs_high = np.array([10., 50.])
            self.max_act = np.array([30.])
        elif self.action_type == 'R': #relative
            self.obs_high = np.array([10., 50., 30.])
            self.max_act = np.array([30.])
        else:
            raise ValueError('Action type for PerlinEnv.AccelerationEnv not recognized. Accepted types are "R" for relative actions and "A" for absolute actions')

        self.observation_space = spaces.Box(low=-self.obs_high, high=self.obs_high)
        self.action_space = spaces.Box(low=-self.max_act, high=self.max_act)

        self.velocity = np.array([0.])
        self.acceleration = np.array([0.])

        print('Initialized with type ' + self.action_type)

    def step(self, u):
        if self.done:
            warnings.warn('Max sim time exceeded. Time to reset')
        
        self.update_goal()
        u = np.clip(u, -self.max_act, self.max_act)

        if self.action_type == 'A':
            self.acceleration = u
        elif self.action_type == 'R':
            self.acceleration += u
        
        self.velocity += self.acceleration * self.dt

        self.state += self.velocity * self.dt
        self.sim_time += self.dt

        if self.sim_time > self.max_time:
            self.done = True
        
        reward = -np.abs(self.goal - self.state)

        state_return = None
        if self.action_type == 'A':
            state_return = np.concatenate((self.goal-self.state, self.velocity))
            # state_return = np.concatenate((self.goal-self.state, self.state, self.velocity, self.acceleration))
        elif self.action_type == 'R':
            state_return = np.concatenate([self.goal-self.state, self.velocity, self.acceleration])

        return state_return, reward, self.done, {'goal':self.goal, 'sim_time':self.sim_time-self.dt}

    def reset(self):
        super().reset()

        self.velocity = np.array([0.])
        self.acceleration = np.array([0.])
        
        state_return = None
        if self.action_type == 'A':
            state_return = np.concatenate((self.goal-self.state, self.velocity))
            # state_return = np.concatenate((self.goal-self.state, self.state, self.velocity, self.acceleration))
        elif self.action_type == 'R':
            state_return = np.concatenate((self.goal-self.state, self.velocity, self.acceleration))

        return state_return