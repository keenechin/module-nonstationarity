from soft_fingers import SoftFingerModules
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class SoftFingerModulesEnv(gym.Env):
    def __init__(self):
        self.hardware = SoftFingerModules()
        self.action_space = spaces.Box(
            low=np.array([self.hardware.min['left'], self.hardware.min['right']] * 3),
            high=np.array([self.hardware.max['left'], self.hardware.max['right']]* 3),
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


    def _get_obs(self):
        all_theta = self.hardware.get_pos()
        theta_joints = all_theta[0:6]
        theta_t_obj = all_theta[6]
        theta_t_obj_sincos = [np.sin(theta_t_obj), np.cos(theta_t_obj)]
        theta_dot_joints = None
        last_action = None
        dtheta_obj = None

        state = np.array([*theta_joints, 
                           *theta_dot_joints,
                           *theta_t_obj_sincos,
                           *last_action,
                           dtheta_obj])
        return state 

    
    def step(self, u):
        pass

    def decompose(self, state):
        theta_joints = state[0:6]
        theta_dot_joints = state[6:12]
        theta_t_obj_sincos = state[12:14]
        last_action = state[14:20]
        dtheta_obj = state[20]
        return theta_joints, theta_dot_joints, theta_t_obj_sincos, \
               last_action, dtheta_obj


    def one_if(self, x, thresh):
        return (0,1)[x<thresh]


    def reward(self, state):
        #rt=−5|∆θt,obj|−‖θnominal−θt‖−∥∥∥ ̇θt∥∥∥+ 101(|∆θt,obj|<0.25) + 501(|∆θt,obj|<0.1
        theta_joints, theta_dot_joints, _, _, dtheta_obj = self.decompose(state)
        return -5 * np.abs(dtheta_obj) \
                -np.linalg.norm(self.hardware.default - theta_joints) \
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
    # env_name = 'CartPole-v1'
    env_name = 'SoftFingerModulesEnv-v0'

    env = gym.make(env_name)