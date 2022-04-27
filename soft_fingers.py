from queue import Empty
from dynamixel_driver import dxl
import numpy as np
import time


class SoftFingerModules():
    def __init__(self, motor_type="X",
                 device="/dev/ttyUSB0", baudrate=57600, protocol=2, operating_torque = 80):

        self.finger1_id = [42, 43]
        self.finger2_id = [40, 41]
        self.finger3_id = [44, 45]
        self.object_id = [50]
        self.servos = dxl(motor_id=[
            *self.finger1_id, *self.finger2_id, *self.finger3_id, *self.object_id],
            motor_type=motor_type,
            devicename=device, baudrate=baudrate, protocol=protocol, operating_torque=operating_torque)
        self.servos.open_port()
        self.mid = np.pi
        range = 2.1
        self.min = {"left": self.mid -  3 * range/4, "right": self.mid - range/4}
        self.max = {"left": self.mid + range/4, "right": self.mid + 3 *range/4}
        self.action_low = np.array([self.min["left"], self.min["right"]]*3)
        self.action_high = np.array([self.max["left"], self.max["right"]]*3)
        # self.action_low = np.array([0])
        # self.action_high = np.array([1])
        # current joint angles
        # the joint velocities,
        # the sine and cosine values of the object’s angle,
        # the last action, the error between the goal and the current object angle
        self.observation_low  = np.array([
            *self.action_low,
            *(self.action_low - self.action_high),
            -1, -1, 
            *self.action_low,
            -2 *np.pi
        ])

        self.observation_high  = np.array([
            *self.action_high,
            *(self.action_high - self.action_low),
            1, 1, 
            *self.action_high,
            2 *np.pi
        ])

        finger_default = (self.min["left"], self.max["right"])
        self.theta_joints_nominal = self.env_action(np.array(finger_default * 3))

        # self.random = np.random.RandomState(118342328)
        self.random = np.random

        time.sleep(0.1)
        self.reset()
    
    def wrap(self, angle):
        """Maps angle in radians to [-1, 1] """
        return np.arctan2(np.sin(angle), np.cos(angle)) / np.pi
    
    def unwrap(self, angle):
        """Maps [-1, 1] to [0, 2pi] """
        return np.mod(angle * np.pi + 2*np.pi, 2*np.pi) 

    def env_action(self, action):
        env_action = -1 + (action - self.action_low) * 2/(self.action_high - self.action_low)
        return env_action

    def hardware_action(self, action):
        hardware_action = self.action_low + (action + 1) * (self.action_high - self.action_low)/2
        return hardware_action

    def env_observation(self, obs):
        env_obs = -1 + (obs - self.observation_low) * 2/(self.observation_high - self.observation_low)
        return env_obs

    def hardware_observation(self, obs):
        hardware_obs = self.observation_low + (obs + 1) * (self.observation_high - self.observation_low)/2
        return hardware_obs

    def reset(self):
        self.servos.engage_motor(self.object_id, True)
        self.hardware_move(self.theta_joints_nominal)
        self.move_object(1)
        self.servos.engage_motor(self.object_id, False)
    

    def move_object(self, pos, err_thresh=0.1):
        self.servos.engage_motor(self.object_id, True)
        errs = np.array([np.inf] * 1)
        self.servos.set_des_pos([self.servos.motor_id[-1]], [self.unwrap(pos)])
        while np.any(errs > err_thresh):
            curr = self.get_pos_obj()
            errs = np.abs(curr - pos)
        self.object_pos = curr[0]
        print(self.object_pos)
        self.servos.engage_motor(self.object_id, False)

    def move_obj_random(self):
        pos = self.random.uniform(-1, 1)
        self.move_object(pos) 

    def finger_delta(self, finger_num, dir, mag=0.3):
        movements = {"up": np.array([mag, -mag]),
                     "down": np.array([-mag, mag]),
                     "left": np.array([mag, mag]),
                     "right": np.array([-mag, -mag])}
        assert finger_num in [0, 1, 2]
        assert dir in movements.keys()
        delta = movements[dir]
        pos = self.get_pos_fingers()
        left = (finger_num)*2
        right = (finger_num)*2+2
        pos[left: right] = pos[left: right] + delta
        action = np.clip(pos, -1, 1)
        return action

    def finger_move(self, finger_num, finger_pos):
        assert finger_num in [0, 1, 2]
        left = (finger_num)*2
        right = (finger_num)*2+2
        pos = self.get_pos_fingers()
        pos[left:right] = finger_pos
        action = np.clip(pos, -1, 1)
        return action

    def hardware_move(self, pos, err_thresh=0.05, derr_thresh=0.1, timeout=0.2):
        pos = self.hardware_action(pos)
        self.servos.set_des_pos(self.servos.motor_id[:-1], pos)
        errs = np.array([np.inf] * 6)
        last_errs = np.array([np.inf] * 6)
        start = time.time()
        while np.any(errs > err_thresh):
            curr = self.get_pos_fingers()
            errs = np.abs(curr - pos)
            derrs = np.abs(last_errs - errs)
            last_errs = errs
            elapsed = time.time() - start
            timedout = elapsed > timeout
            finger_err = np.empty((3,2))
            not_improving = np.empty((3,1))
            for i in range(3):
                finger_err[i,:] = derrs[2*i: 2*i+2]
                not_improving[i] = np.all(finger_err[i,:] < [derr_thresh]*2)
                
            if  timedout and np.any(not_improving):
                break 
        self.object_pos = self.get_pos_obj()[0]
            
    def get_pos_all(self):
        pos =  [*self.get_pos_fingers(), *self.get_pos_obj()]
        return pos
    
    def get_pos_fingers(self):
        return self.env_action(self.servos.get_pos(self.servos.motor_id[:-1]))

    def get_pos_obj(self):
        return self.wrap(self.servos.get_pos([self.servos.motor_id[-1]]))

    def __del__(self):
        self.servos.engage_motor(self.servos.motor_id, False)








if __name__ == "__main__":
    from manual_controller import ExpertActionStream
    manipulator = SoftFingerModules()
    with ExpertActionStream(manipulator, target_theta=0.0) as action_listener:
        obj, process, action_channel, state_channel = action_listener
        
        while process.is_alive():
            try:
                command = action_channel.get(False)
                try:
                    pos = obj.parse(command)
                    manipulator.hardware_move(pos)
                    while not state_channel.empty():
                        state_channel.get()
                    state_channel.put(manipulator.object_pos)
                except TypeError:
                    pass
            except Empty:
                pass

    manipulator.reset()
