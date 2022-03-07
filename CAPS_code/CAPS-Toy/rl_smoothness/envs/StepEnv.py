import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import warnings

class BaseEnv():
    def __init__(self, dt=0.02, max_time=3., decaying_ac=None, action_type='R', seed=None, **kwargs):
        self.sim_time = 0.
        self.max_time = max_time
        self.dt = dt
        self.decaying_ac = decaying_ac
        
        self.last_action = np.array([0])
        self.a_decay = 0.
        
        self.action_type = action_type
        self.seed(seed)

        # Dummy variable to be overwritten later
        self.obs_high = np.array([1.])

        self.goal = (2*self.np_random.rand(1) - 1)*self.obs_high[0]
        self.goal_set = False
        self.state = self.goal.copy()

        self.done = False


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def update_goal(self):
        if self.sim_time > self.max_time/3 and not self.goal_set:
            self.goal = (2*self.np_random.rand(1) - 1)*self.obs_high[0]
            self.goal_set = True

    def reset(self):
        self.sim_time = 0.
        self.goal = (2*self.np_random.rand(1) - 1)*self.obs_high[0]
        self.goal_set = False
        self.state = self.goal.copy()
        self.done = False

class StateEnv(BaseEnv):
    def __init__(self, dt=0.02, max_time=3., decaying_ac=None, action_type='R', seed=None, **kwargs):
        super().__init__(dt=dt, max_time=max_time, decaying_ac=decaying_ac, action_type=action_type, seed=seed, **kwargs)

        if self.action_type == 'A': #absolute
            obs_high = [10., 10.]
            max_act = [10.]
        elif self.action_type == 'R': #relative
            obs_high = [10.]
            max_act = [1.]
        else:
            raise ValueError('Action type for StepEnv not recognized. Accepted types are "R" for relative actions and "A" for absolute actions')

        if self.decaying_ac:
            obs_high.append(10.)

        self.obs_high = np.array(obs_high)
        self.max_act = np.array(max_act)

        self.action_space = spaces.Box(low=-self.max_act, high=self.max_act)
        self.observation_space = spaces.Box(low=-self.obs_high, high=self.obs_high)

        self.goal = (2*self.np_random.rand(1) - 1)*self.obs_high[0]
        self.goal_set = False
        self.state = self.goal.copy()

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

        reward = -np.abs(self.goal - self.state)

        if self.decaying_ac:
            self.a_decay *= 0.995
            self.a_decay += (self.a_decay*0.001+1)*(u - self.last_action[0])
            reward -= self.a_decay

        self.last_action = np.array([u])

        state_return = None
        if self.action_type == 'A':
            state_return = np.concatenate((self.goal-self.state, self.state))
        elif self.action_type == 'R':
            state_return = self.goal-self.state

        if self.decaying_ac:
            print(state_return)
            print(self.last_action)
            state_return = np.concatenate((state_return, self.last_action))

        return state_return, reward, self.done, {'goal':self.goal, 'sim_time':self.sim_time-self.dt}

    def reset(self):
        super().reset()
        self.last_action = np.array([0])
        self.a_decay = 0.

        state_return = None
        if self.action_type == 'A':
            state_return = np.concatenate((self.goal-self.state, self.state))
        elif self.action_type == 'R':
            state_return = self.goal-self.state

        if self.decaying_ac:
            state_return = np.concatenate((np.array(state_return), np.array(self.last_action)))

        return state_return

class VelocityEnv(BaseEnv):
    def __init__(self, dt=0.02, max_time=3., decaying_ac=None, action_type='R', seed=None, **kwargs):
        super().__init__(dt=dt, max_time=max_time, decaying_ac=decaying_ac, action_type=action_type, seed=seed, **kwargs)

        if self.action_type == 'A': #absolute
            obs_high = [10.]
            max_act = [50.]
        elif self.action_type == 'R': #relative
            obs_high = [10., 50.]
            max_act = [50.]
        else:
            raise ValueError('Action type for StepEnv not recognized. Accepted types are "R" for relative actions and "A" for absolute actions')

        if self.decaying_ac:
            obs_high.append(5.)

        self.obs_high = np.array(obs_high)
        self.max_act = np.array(max_act)

        self.action_space = spaces.Box(low=-self.max_act, high=self.max_act)
        self.observation_space = spaces.Box(low=-self.obs_high, high=self.obs_high)

        self.goal = (2*self.np_random.rand(1) - 1)*self.obs_high[0]
        self.goal_set = False
        self.state = self.goal.copy()
        self.velocity = np.array([0.])

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

        if self.decaying_ac:
            self.a_decay *= 0.995
            self.a_decay += (self.a_decay*0.001+1)*(u - self.last_action[0])
            reward -= self.a_decay

        self.last_action = np.array([u])

        state_return = None
        if self.action_type == 'A':
            state_return = self.goal-self.state
        elif self.action_type == 'R':
            state_return = np.concatenate((self.goal-self.state, self.velocity))

        if self.decaying_ac:
            print(state_return)
            print(self.last_action)
            state_return = np.concatenate((state_return, self.last_action))

        return state_return, reward, self.done, {'goal':self.goal, 'sim_time':self.sim_time-self.dt}

    def reset(self):
        super().reset()
        self.last_action = np.array([0])
        self.a_decay = 0.

        self.velocity = np.array([0.])

        state_return = None
        if self.action_type == 'A':
            state_return = self.goal-self.state
            # state_return = np.concatenate((self.goal-self.state, self.state, self.velocity))
        elif self.action_type == 'R':
            state_return = np.concatenate((self.goal-self.state, self.velocity))

        if self.decaying_ac:
            state_return = np.concatenate((np.array(state_return), np.array(self.last_action)))

        return state_return

class AccelerationEnv(BaseEnv):
    def __init__(self, dt=0.02, max_time=3., decaying_ac=None, action_type='R', seed=None, **kwargs):
        super().__init__(dt=dt, max_time=max_time, decaying_ac=decaying_ac, action_type=action_type, seed=seed, **kwargs)

        if self.action_type == 'A': #absolute
            obs_high = [10., 50.]
            max_act = [50.]
        elif self.action_type == 'R': #relative
            obs_high = [10., 5., 3.]
            max_act = [1.]
        else:
            raise ValueError('Action type for StepEnv not recognized. Accepted types are "R" for relative actions and "A" for absolute actions')

        if self.decaying_ac:
            obs_high.append(5.)

        self.obs_high = np.array(obs_high)
        self.max_act = np.array(max_act)

        self.action_space = spaces.Box(low=-self.max_act, high=self.max_act)
        self.observation_space = spaces.Box(low=-self.obs_high, high=self.obs_high)

        self.goal = (2*self.np_random.rand(1) - 1)*self.obs_high[0]
        self.goal_set = False
        self.state = self.goal.copy()
        self.velocity = np.array([0.])
        self.acceleration = np.array([0.])

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

        if self.decaying_ac:
            self.a_decay *= 0.995
            self.a_decay += (self.a_decay*0.001+1)*(u - self.last_action[0])
            reward -= self.a_decay

        self.last_action = np.array([u])

        state_return = None
        if self.action_type == 'A':
            state_return = np.concatenate((self.goal-self.state, self.velocity))
        elif self.action_type == 'R':
            state_return = np.concatenate((self.goal-self.state, self.velocity, self.acceleration))

        if self.decaying_ac:
            print(state_return)
            print(self.last_action)
            state_return = np.concatenate((state_return, self.last_action))

        return state_return, reward, self.done, {'goal':self.goal, 'sim_time':self.sim_time-self.dt}

    def reset(self):
        super().reset()
        self.last_action = np.array([0])
        self.a_decay = 0.

        self.velocity = np.array([0.])
        self.acceleration = np.array([0.])

        state_return = None
        if self.action_type == 'A':
            state_return = np.concatenate((self.goal-self.state, self.velocity))
        elif self.action_type == 'R':
            state_return = np.concatenate((self.goal-self.state, self.velocity, self.acceleration))

        if self.decaying_ac:
            state_return = np.concatenate((np.array(state_return), np.array(self.last_action)))

        return state_return