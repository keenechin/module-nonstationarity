from gym import Env

class LearnedDynamicsEnv(Env):
    def __init__(self, hardware_env, dynamics):
        self.action_space = hardware_env.action_space
        self.observation_space = hardware_env.observation_space
        self.action_size = self.action_space.shape[0]
        self.observation_size = self.observation_space.shape[0]
        self.dynamics = dynamics




