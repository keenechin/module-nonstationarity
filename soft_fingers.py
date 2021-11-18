from dynamixel_driver import dxl
import numpy as np
import time
from queue import Queue
from pynput import keyboard


class SoftFingerModules():
    def __init__(self, ids=[40, 41, 42, 43, 44, 45, 50], motor_type="X",
                 device="/dev/ttyUSB1", baudrate=57600, protocol=2):
        self.servos = dxl(motor_id=ids, motor_type=motor_type,
                          devicename=device, baudrate=baudrate, protocol=protocol)
        self.servos.open_port()
        self.mid = np.pi
        range = 2.75
        self.min = {"left": self.mid - range/2, "right": self.mid + range/2}
        self.max = {"left": self.mid + range/4, "right": self.mid - range/4}

        self.reset()

    def reset(self):
        default = ((self.min["left"], self.min["right"]),) * 3
        self.all_move(default)
        self.servos.set_des_pos([self.servos.motor_id[-1]], [self.mid])
        err_thresh = 0.05
        errs = np.array([np.inf] * 1)
        while np.any(errs > err_thresh):
            curr = self.get_pos()
            errs = np.abs(curr[-1] - np.pi)
        self.servos.engage_motor([50], False)

    def finger_delta(self, finger_num, dir):
        movements = {"up": np.array([0.1, -0.1]),
                     "down": np.array([-0.1, 0.1]),
                     "left": np.array([0.1, 0.1]),
                     "right": np.array([-0.1, -0.1])}
        assert dir in movements.keys()
        assert finger_num in [0, 1, 2]
        curr = self.get_pos()
        left = (finger_num)*2
        right = (finger_num)*2+1
        pos = np.array(curr[left:right+1])
        delta = movements[dir]
        new_pos = pos + delta
        self.finger_move(finger_num, new_pos)

    def finger_stop(self, finger_num):
        curr = self.get_pos()
        left = (finger_num)*2
        right = (finger_num)*2+1
        self.finger_move(finger_num, curr[left:right+1])

    def finger_move(self, finger_num, pos, err_thresh=0.1):
        assert finger_num in [0, 1, 2]
        left = (finger_num)*2
        right = (finger_num)*2+1
        self.servos.set_des_pos(self.servos.motor_id[left:right+1], pos)
        errs = np.array([np.inf, np.inf])
        while np.any(errs > err_thresh):
            curr = self.get_pos()
            errs = np.abs(curr[left:right+1] - pos)

    def all_move(self, pos, err_thresh=0.1):
        for i in range(3):
            self.finger_move(i, pos[i])

    def get_pos(self):
        return self.servos.get_pos(self.servos.motor_id)


def get_listener_funcs(queue):
    def on_press(key):

        command = lambda: None
        try:
            if key.char == 'w':
                command = lambda: manipulator.finger_delta(0, 'up')
            if key.char == 's':
                command = lambda: manipulator.finger_delta(0, 'down')
            if key.char == 'a':
                command = lambda: manipulator.finger_delta(0, 'left')
            if key.char == 'd':
                command = lambda: manipulator.finger_delta(0, 'right')
        except AttributeError:
            pass

        try:
            if key.char == 'i':
                command = lambda: manipulator.finger_delta(1, 'up')
            if key.char == 'k':
                command = lambda: manipulator.finger_delta(1, 'down')
            if key.char == 'j':
                command = lambda: manipulator.finger_delta(1, 'left')
            if key.char == 'l':
                command = lambda: manipulator.finger_delta(1, 'right')
        except AttributeError:
            pass

        if key == keyboard.Key.up:
            command = lambda: manipulator.finger_delta(2, 'up')
        if key == keyboard.Key.down:
            command = lambda: manipulator.finger_delta(2, 'down')
        if key == keyboard.Key.left:
            command = lambda: manipulator.finger_delta(2, 'left')
        if key == keyboard.Key.right:
            command = lambda: manipulator.finger_delta(2, 'right')

        while not queue.empty():
           queue.get() 
        queue.put(command)

    def on_release(key):
        try:
            if key == keyboard.Key.esc or key.char == 'q':
                return False
        except AttributeError:
            pass

    return on_press, on_release


def start_user_input():
    queue = Queue()
    on_press, on_release = get_listener_funcs(queue)
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    return listener, queue


if __name__ == "__main__":
    manipulator = SoftFingerModules()
    listener,queue = start_user_input()
    for i in range(int(1e6)):
        command = queue.get()
        command()
    listener.join()
    manipulator.reset()
