from queue import Empty
from dynamixel_driver import dxl
import numpy as np
import time
from multiprocessing import Process, Queue
import pygame


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
        self.hardware_move(self.theta_joints_nominal)
        self.move_obj_random()

    def move_object(self, pos, err_thresh=0.1):
        print(pos)
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

    def finger_delta(self, finger_num, dir, mag=0.15):
        movements = {"up": np.array([mag, -mag]),
                     "down": np.array([-mag, mag]),
                     "left": np.array([mag, mag]),
                     "right": np.array([-mag, -mag])}
        assert dir in movements.keys()
        assert finger_num in [0, 1, 2]
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
        print(pos)
        return pos
    
    def get_pos_fingers(self):
        return self.env_action(self.servos.get_pos(self.servos.motor_id[:-1]))

    def get_pos_obj(self):
        return self.wrap(self.servos.get_pos([self.servos.motor_id[-1]]))





class ExpertActionStream():
    def __init__(self, manipulator, target_theta):
        self.manipulator = manipulator
        self.target_theta = target_theta
        self.func_names = {'move': manipulator.finger_move,
                    'delta': manipulator.finger_delta,
                    'idle': manipulator.get_pos_fingers}

    def parse(self, command):
        func_name = command['func']
        assert func_name in self.func_names
        func = self.func_names[func_name]
        params = command['params']
        if params is not None:
            action = func(*params)
        else:
            action = func()
        return action


    def get_listener_funcs(self, queue):
        def on_press(key):
            command = {'func': 'idle', 'params': None}

            if key == pygame.K_4:
                command = {'func': 'idle', 'params': None}

            try:
                if key == pygame.K_1:
                    command = {'func': 'move', 'params': (
                        0, self.manipulator.theta_joints_nominal[0:2])}
                elif key == pygame.K_2:
                    command = {'func': 'move', 'params': (
                        1, self.manipulator.theta_joints_nominal[2:4])}
                elif key == pygame.K_3:
                    command = {'func': 'move', 'params': (
                        2, self.manipulator.theta_joints_nominal[4:6])}
            except AttributeError:
                pass

            try:
                if key == pygame.K_w:
                    command = {'func': 'delta', 'params': (0, 'up')}
                elif key == pygame.K_s:
                    command = {'func': 'delta', 'params': (0, 'down')}
                elif key == pygame.K_a:
                    command = {'func': 'delta', 'params': (0, 'left')}
                elif key == pygame.K_d:
                    command = {'func': 'delta', 'params': (0, 'right')}
            except AttributeError:
                pass

            try:
                if key == pygame.K_i:
                    command = {'func': 'delta', 'params': (1, 'up')}
                elif key == pygame.K_k:
                    command = {'func': 'delta', 'params': (1, 'down')}
                elif key == pygame.K_j:
                    command = {'func': 'delta', 'params': (1, 'left')}
                elif key == pygame.K_l:
                    command = {'func': 'delta', 'params': (1, 'right')}
            except AttributeError:
                pass

            if key == pygame.K_UP:
                command = {'func': 'delta', 'params': (2, 'up')}
            elif key == pygame.K_DOWN:
                command = {'func': 'delta', 'params': (2, 'down')}
            elif key == pygame.K_LEFT:
                command = {'func': 'delta', 'params': (2, 'left')}
            elif key == pygame.K_RIGHT:
                command = {'func': 'delta', 'params': (2, 'right')}

            while not queue.empty():
                queue.get()
            queue.put(command)

            return True

        def on_release(key):
            try:
                if key == pygame.K_ESCAPE or key == pygame.K_q:
                    return False
                else:
                    return True
            except AttributeError:
                pass


        return on_press, on_release


    def listen_keys(self, on_press, on_release):
        pygame.init()
        width = 960
        height = 540
        white = (255, 255, 255)
        green = (10, 70, 0)
        blue = (200, 200, 255)
        window = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Soft Finger Keyboard Controller')
        pygame.display.set_icon(pygame.image.load('soft_finger_icon.png'))
        target_pic = pygame.image.load('target_position.jpg').convert()
        old_width, old_height = target_pic.get_size()
        pic_width  = width//3
        pic_height = old_height * pic_width // old_width 
        target_pic = pygame.transform.scale(target_pic, (pic_width, pic_height))
        fontsize = 32
        font = pygame.font.Font('freesansbold.ttf', fontsize)
        text1 = font.render("Finger 1: Move with w, a, s, d , Retract with 1", True, green, blue).convert()
        text2 = font.render("Finger 2: Move with i, j, k, l , Retract with 2", True, green, blue).convert()
        text3 = font.render("Finger 3: Move with up, down, left, right,  Retract with 3", True, green, blue).convert()
        text_quit = font.render("Quit with Esc, or q", True, (255, 0, 0), (0,0,0)).convert()
        textRect1 = text1.get_rect()
        textRect1 = (0, fontsize)
        textRect2 = text2.get_rect()
        textRect2 = (0, 2 * fontsize)
        textRect3 = text3.get_rect()
        textRect3 = (0, 3*fontsize)
        quitRect = text_quit.get_rect()
        quitRect = (0,0)


        current_angle = np.around(self.manipulator.object_pos, 3)
        clock = pygame.time.Clock()
        run = True
        while run:
            clock.tick(5)
            try:
                current_angle = self.state_channel.get_nowait()
                current_angle = np.around(current_angle, 3)
            except Empty:
                pass
            target_angle = np.around(self.target_theta, 3)
            current_error = current_angle - target_angle
            if current_angle > 0:
                plus = '+'
            text_current = font.render(f"Current angular error: {current_error}\u03C0 rad", True, white, (0,0,0))
            text_target = font.render(f"                Target angle: {target_angle}\u03C0 rad", True, white, (0,0,0))
            targetRect = text_target.get_rect()
            currentRect = text_current.get_rect()
            targetRect = (width//10, height//2 + fontsize) 
            currentRect = (width//10, height//2)
            window.fill(white)
            window.blit(text_target, targetRect)
            window.blit(text_current, currentRect)
            window.blit(text1, textRect1)
            window.blit(text2, textRect2)
            window.blit(text3, textRect3)
            window.blit(text_quit, quitRect)
            window.blit(target_pic, ((width*2)//3, height//3))
            
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                elif event.type == pygame.KEYDOWN:
                    run = on_press(event.key)
                elif event.type == pygame.KEYUP:
                    run = on_release(event.key)



    def __enter__(self):
        self.action_channel = Queue()
        self.state_channel = Queue()
        on_press, on_release = self.get_listener_funcs(self.action_channel)
        self.listener_process = Process(target=self.listen_keys, args=(on_press, on_release))
        self.listener_process.start()
        print("Starting keyboard listener.")
        return self, self.listener_process, self.action_channel, self.state_channel

    def __exit__(self, exc_type, exc_value, exc_traceback):
        print("Closing keyboard listener.")
        self.listener_process.terminate()



if __name__ == "__main__":
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
