from soft_fingers import SoftFingerModules, ExpertActionStream
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class SoftFingerModulesEnv(gym.Env):
    def __init__(self, nominal_theta = 0):
        self.hardware = SoftFingerModules()
        low = [self.hardware.min['left'], self.hardware.min['right']] * 3
        high = [self.hardware.max['left'], self.hardware.max['right']]* 3
        self.action_space = spaces.Box(
            low=np.array(low),
            high=np.array(high),
            dtype=np.float64
        )

        # current joint angles
        # the joint velocities,
        # the sine and cosine values of the object’s angle,
        # the last action, the error between the goal and the current object angle
        # 
        self.observation_space = spaces.Box(
            low=np.array([
                *[self.hardware.min['left'], self.hardware.min['right']] * 3,
                *[-2 * np.pi] * 6,
                -1, -1,
                *[self.hardware.min['left'], self.hardware.min['right']] * 3,
                -2 *np.pi]),
            high=np.array([
                *[self.hardware.max['left'], self.hardware.max['right']] * 3,
                *[2 * np.pi] * 6,
                1, 1,
                *[self.hardware.max['left'], self.hardware.max['right']] * 3,
                2 * np.pi]),
            dtype=np.float64
        )

        self.action_size = self.action_space.shape[0]
        self.observation_size = self.observation_space.shape[0]

        self.last_action = self.hardware.theta_joints_nominal
        self.last_pos = self.hardware.theta_joints_nominal
        self.nominal_theta = nominal_theta


    def _get_obs(self):
        all_theta = self.hardware.get_pos_all()
        theta_joints = all_theta[0:6]
        theta_t_obj = all_theta[6]
        theta_t_obj_sincos = [np.sin(theta_t_obj), np.cos(theta_t_obj)]
        theta_dot_joints = theta_joints - self.last_pos # TODO: Add instantaneous velocity
        last_action = self.last_action
        dtheta_obj = theta_t_obj - self.nominal_theta

        state = np.array([*theta_joints, 
                           *theta_dot_joints,
                           *theta_t_obj_sincos,
                           *last_action,
                           dtheta_obj])
        return state 

    
    def step(self, action):
        self.hardware.all_move(action)
        self.last_action = action
        self.last_pos = self.hardware.get_pos_fingers()
        state = self._get_obs()
        reward = self.reward(state)
        return state, reward, False, {}

    def decompose(self, state):
        theta_joints = state[0:6]
        theta_dot_joints = state[6:12]
        theta_t_obj_sincos = state[12:14]
        last_action = state[14:20]
        dtheta_obj = state[20]
        return theta_joints, theta_dot_joints, theta_t_obj_sincos, \
               last_action, dtheta_obj


    def one_if(self, x, thresh):
        # return 1 if x < thresh, 0 if not 
        return (0,1)[x<thresh]


    def reward(self, state):
        #rt=−5|∆θt,obj|−‖θnominal−θt‖−∥∥∥ ̇θt∥∥∥+ 101(|∆θt,obj|<0.25) + 501(|∆θt,obj|<0.1
        theta_joints, theta_dot_joints, _, _, dtheta_obj = self.decompose(state)
        return -5 * np.abs(dtheta_obj) \
                -np.linalg.norm(self.hardware.theta_joints_nominal - theta_joints) \
                -np.linalg.norm(theta_dot_joints) \
                + 10 * self.one_if(np.abs(dtheta_obj), thresh=0.25) \
                + 50 * self.one_if(np.abs(dtheta_obj), thresh=0.10)

    def reset(self):
        self.hardware.reset()
        return self._get_obs()


if __name__ == "__main__":
    gym.envs.register(
        id='SoftFingerModulesEnv-v0',
        entry_point='module_env:SoftFingerModulesEnv',
        kwargs={}
    )
    import time
    env_name = 'SoftFingerModulesEnv-v0'
    env = gym.make(env_name)
    for i in range(100):
        action = env.action_space.sample()
        env.step(action)
        print(env.last_pos - action)
        input("Press any key to continue.")


    env.reset()


