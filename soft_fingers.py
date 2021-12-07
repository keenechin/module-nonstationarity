from queue import Empty
from dynamixel_driver import dxl
import numpy as np
import time
from multiprocessing import Process, Queue
import pygame


class SoftFingerModules():
    def __init__(self, motor_type="X",
                 device="/dev/ttyUSB0", baudrate=57600, protocol=2):

        self.finger1_id = [40, 41]
        self.finger2_id = [42, 43]
        self.finger3_id = [44, 45]
        self.object_id = [50]
        self.servos = dxl(motor_id=[
            *self.finger1_id, *self.finger2_id, *self.finger3_id, *self.object_id],
            motor_type=motor_type,
            devicename=device, baudrate=baudrate, protocol=protocol)
        self.servos.open_port()
        self.mid = np.pi
        range = 2.1
        self.min = {"left": self.mid -  3 * range/4, "right": self.mid - range/4}
        self.max = {"left": self.mid + range/4, "right": self.mid + 3 *range/4}
        self.finger_default = (self.min["left"], self.max["right"])
        self.theta_joints_nominal = np.array(self.finger_default * 3)
        self.theta_obj_nominal = np.pi
        # self.theta_joints_nominal = np.array([self.mid] * 6)

        self.func_names = {'move': self.finger_move,
                           'delta': self.finger_delta,
                           'idle': self.get_pos_fingers}
        time.sleep(0.1)
        self.reset()

    def reset(self):
        self.all_move(self.theta_joints_nominal)
        self.servos.engage_motor(self.object_id, True)
        self.move_object(self.mid)
        self.servos.engage_motor(self.object_id, False)

    def move_object(self, pos, err_thresh=0.01):
        errs = np.array([np.inf] * 1)
        self.servos.set_des_pos([self.servos.motor_id[-1]], [pos])
        while np.any(errs > err_thresh):
            curr = self.get_pos_obj()
            errs = np.abs(curr - self.theta_obj_nominal)
        self.object_pos = curr[0]

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
        return pos

    def finger_move(self, finger_num, finger_pos):
        assert finger_num in [0, 1, 2]
        left = (finger_num)*2
        right = (finger_num)*2+2
        pos = self.get_pos_fingers()
        pos[left:right] = finger_pos
        return pos

    def all_move(self, pos, err_thresh=0.05, timeout=0.3):
        for i in range(len(pos)): 
            if i%2 == 0:
                pos[i] = np.clip(pos[i], self.min['left'], self.max['left'])
            else:
                pos[i] = np.clip(pos[i], self.min['right'], self.max['right'])
        self.servos.set_des_pos(self.servos.motor_id[:-1], pos)
        errs = np.array([np.inf] * 6)
        start = time.time()
        while np.any(errs > err_thresh):
            curr = self.get_pos_fingers()
            errs = np.abs(curr - pos)
            elapsed = time.time() - start
            if elapsed > timeout:
                break 
        self.object_pos = self.get_pos_obj()[0]
            
    def get_pos_all(self):
        return self.servos.get_pos(self.servos.motor_id)
    
    def get_pos_fingers(self):
        return self.servos.get_pos(self.servos.motor_id[:-1])

    def get_pos_obj(self):
        return self.servos.get_pos([self.servos.motor_id[-1]])

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





class ExpertActionStream():
    def __init__(self, manipulator, target_theta):
        self.manipulator = manipulator
        self.target_theta = target_theta


    def get_listener_funcs(self, queue):
        def on_press(key):
            command = {'func': 'idle', 'params': None}

            if key == pygame.K_4:
                command = {'func': 'idle', 'params': None}

            try:
                if key == pygame.K_1:
                    command = {'func': 'move', 'params': (
                        0, self.manipulator.finger_default)}
                elif key == pygame.K_2:
                    command = {'func': 'move', 'params': (
                        1, self.manipulator.finger_default)}
                elif key == pygame.K_3:
                    command = {'func': 'move', 'params': (
                        2, self.manipulator.finger_default)}
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
        width = 1920
        height = 1080
        white = (255, 255, 255)
        green = (10, 70, 0)
        blue = (200, 200, 255)
        window = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Soft Finger Keyboard Controller')
        pygame.display.set_icon(pygame.image.load('soft_finger_icon.png'))
        fontsize = 32
        font = pygame.font.Font('freesansbold.ttf', fontsize)
        text1 = font.render("Finger 1: Move with w, a, s, d , Retract with 1", True, green, blue)
        text2 = font.render("Finger 2: Move with i, j, k, l , Retract with 2", True, green, blue)
        text3 = font.render("Finger 3: Move with up, down, left, right,  Retract with 3", True, green, blue)
        text_quit = font.render("Quit with Esc, or q", True, (255, 0, 0), (0,0,0))
        textRect1 = text1.get_rect()
        textRect1 = (0, fontsize)
        textRect2 = text2.get_rect()
        textRect2 = (0, 2 * fontsize)
        textRect3 = text3.get_rect()
        textRect3 = (0, 3*fontsize)
        quitRect = text_quit.get_rect()
        quitRect = (0,0)


        current_error = self.manipulator.object_pos
        clock = pygame.time.Clock()
        run = True
        while run:
            clock.tick(5)
            try:
                current_error = self.state_channel.get_nowait()
            except Empty:
                pass
            target_angle = self.target_theta

            if current_error > 0:
                plus = '+'
            text_current = font.render(f"Current angular error: {plus} {current_error} rad", True, white, (0,0,0))
            text_target = font.render(f"                Target angle: {target_angle} rad", True, white, (0,0,0))
            targetRect = text_target.get_rect()
            currentRect = text_current.get_rect()
            targetRect = (width//3, height//2 + fontsize) 
            currentRect = (width//3, height//2)
            window.fill(white)
            window.blit(text_target, targetRect)
            window.blit(text_current, currentRect)
            window.blit(text1, textRect1)
            window.blit(text2, textRect2)
            window.blit(text3, textRect3)
            window.blit(text_quit, quitRect)
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
        return self.listener_process, self.action_channel, self.state_channel

    def __exit__(self, exc_type, exc_value, exc_traceback):
        print("Closing keyboard listener.")
        self.listener_process.terminate()



if __name__ == "__main__":
    manipulator = SoftFingerModules()
    with ExpertActionStream(manipulator, 0) as action_listener:
        process, action_channel, state_channel = action_listener
        
        while process.is_alive():
            try:
                command = action_channel.get(False)
                try:
                    pos = manipulator.parse(command)
                    manipulator.all_move(pos)
                    while not state_channel.empty():
                        state_channel.get()
                    state_channel.put(manipulator.object_pos)
                except TypeError:
                    pass
            except Empty:
                pass

    manipulator.reset()
