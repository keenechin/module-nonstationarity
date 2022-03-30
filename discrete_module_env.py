from soft_fingers import SoftFingerModules
import gym
from gym import spaces
import numpy as np

class DiscreteModuleEnv(gym.Env):
    def __init__(self):
        self.hardware = SoftFingerModules()
        self.action_space = spaces.Discrete(12)
        low = [-1.0]
        high = [1.0]
        self.observation_space = spaces.Box(
            low=np.array(low * 21),
            high=np.array(high * 21),
            dtype=np.float64
        )
        self.action_map = []
        for finger in range(3):
            for dir in ("up", "down", "left", "right"):
                self.action_map.append((finger, dir))
        self.action_map = tuple(self.action_map)
        self.action_size = self.action_space.shape
        self.observation_size = self.observation_space.shape[0]
        self.last_action = 0
        self.last_pos = self.theta_joints_nominal
        

    def _get_obs(self):
        all_theta = self.hardware.get_pos_all()
        theta_joints = all_theta[0:6]
        theta_t_obj = all_theta[6]
        theta_t_obj_sincos = [np.sin(theta_t_obj), np.cos(theta_t_obj)]
        theta_dot_joints = theta_joints - self.last_pos # TODO: Add instantaneous velocity
        last_action = self.last_action
        dtheta_obj = theta_t_obj - self.goal_theta
        state = np.array([*theta_joints, 
                           *theta_dot_joints,
                           *theta_t_obj_sincos,
                           *last_action,
                           dtheta_obj])
        # print(f"State: {state}")
        return state

    def step(self, action, thresh = 0.1/np.pi):
        finger, dir = self.action_map[action]
        action = self.hardware.finger_delta(finger, dir)
        self.hardware.hardware_move(action)
        self.last_action = action
        self.last_pos = self.get_pos_fingers()

    @property
    def theta_joints_nominal(self):
        return self.hardware.theta_joints_nominal

if __name__ == "__main__":
    env = DiscreteModuleEnv()
    import time
    for i in range(12):
        env.step(i)
        time.sleep(1)


