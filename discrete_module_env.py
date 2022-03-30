from soft_fingers import SoftFingerModules
import gym
from gym import spaces
import numpy as np

class DiscreteModuleEnv(gym.Env):
    def __init__(self, goal_theta=0):
        self.goal_theta = goal_theta
        self.hardware = SoftFingerModules()
        self.action_space = spaces.Discrete(15)
        low = [-1.0]
        high = [1.0]
        self.observation_space = spaces.Box(
            low=np.array(low * 16),
            high=np.array(high *16),
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
                           last_action,
                           dtheta_obj])
        # print(f"State: {state}")
        return state

    def step(self, action, thresh = 0.1/np.pi):
        self.last_action = action/14
        if action < 12:
            finger, dir = self.action_map[action]
            action = self.hardware.finger_delta(finger, dir)
        else:
            finger = action-12
            action = self.hardware.finger_move(finger, 
                     self.theta_joints_nominal[finger *2 : (finger+1)*2]) 
        self.hardware.hardware_move(action)
        self.last_pos = self.get_pos_fingers()
        state = self._get_obs()
        reward = self.reward(state)
        err = np.abs(self.object_pos - self.goal_theta)
        done = err < thresh
        return state, reward, done, {}


    def decompose(self, state):
        theta_joints = state[0:6]
        theta_dot_joints = state[6:12]
        theta_t_obj_sincos = state[12:14]
        last_action = state[14]
        dtheta_obj = state[15]
        return theta_joints, theta_dot_joints, theta_t_obj_sincos, \
               last_action, dtheta_obj


    def one_if(self, x, thresh):
        # return 1 if x < thresh, 0 if not 
        if x < thresh:
            return 1
        else:
            return 0


    def reward(self, state):
        #rt=−5|∆θt,obj|−‖θnominal−θt‖−∥∥∥ ̇θt∥∥∥+ 10*1(|∆θt,obj|<0.25) + 50*1(|∆θt,obj|<0.1
        theta_joints, theta_dot_joints, _, _, dtheta_obj = self.decompose(state)
        return -5 * np.abs(dtheta_obj) \
                -0*np.linalg.norm(self.theta_joints_nominal - theta_joints) \
                -0*np.linalg.norm(theta_dot_joints) \
                + 10 * self.one_if(np.abs(dtheta_obj), thresh=0.25/np.pi) \
                + 50 * self.one_if(np.abs(dtheta_obj), thresh=0.10/np.pi)

    @property
    def object_pos(self):
        return self.hardware.get_pos_obj()[0]
        
    @property
    def theta_joints_nominal(self):
        return self.hardware.theta_joints_nominal

    def finger_move(self, finger_num, finger_pos):
        return self.hardware.finger_move(finger_num, finger_pos)

    def finger_delta(self, finger_num, dir):
        return self.hardware.finger_delta(finger_num, dir)

    def get_pos_fingers(self):
        return self.hardware.get_pos_fingers()
        
    def reset(self):
        self.steps = 0
        self.hardware.reset()
        self.hardware.move_obj_random()
        return self._get_obs()

if __name__ == "__main__":
    env = DiscreteModuleEnv()
    import time
    for i in range(15):
        env.step(i)
        env.step(i)
        env.step(i)
        time.sleep(0.1)


