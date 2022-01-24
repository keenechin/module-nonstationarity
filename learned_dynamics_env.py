from gym import Env
import numpy as np
import copy
import torch

class LearnedDynamicsEnv(Env):
    def __init__(self, hardware_env, dynamics):
        self.action_space = hardware_env.action_space
        self.observation_space = hardware_env.observation_space
        self.action_size = self.action_space.shape[0]
        self.observation_size = self.observation_space.shape[0]
        self.dynamics = dynamics
        self.start_state = hardware_env.reset()
        self.state = self.start_state
        self.hardware_env = hardware_env
        print(self.state)
    
    def reset(self):
        self.state = self.start_state
        return self._get_obs()
    
    def _get_obs(self):
        return self.state

    def reward(self, state):
        return self.hardware_env.reward(state)
        
    def step(self, action):
        state = copy.deepcopy(self.state)
        X = np.hstack((state, action))
        state = self.dynamics.predict(torch.from_numpy(X).type(torch.FloatTensor))
        state = state.detach()
        reward = self.reward(self.state)
        return state, reward, False, {}







