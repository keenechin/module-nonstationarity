import pygame
from multiprocessing import Process, Queue
from queue import Empty
import numpy as np

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

