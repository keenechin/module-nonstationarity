import primitives_module_env
import gym
from stable_baselines3 import PPO
import tkinter as tk
import time
import numpy as np

tk.Tk().withdraw()
env =  gym.make(primitives_module_env.env_name)
policy_fname = tk.filedialog.askopenfilename()
policy = PPO.load(policy_fname)
obs = env.reset()
num_steps = 0
# print(f"Starting at {obs}")
start = time.time()
with open(f"learned_task_completion_{primitives_module_env.env_name}.txt", 'w') as f:
    rewards = []
    for i in range(int(input("Enter num trials:"))):
        while True:
            num_steps += 1
            action = policy.predict(obs)[0]
            print(action)
            # action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done or (time.time()-start)>3600:
                print(f"Finished in {num_steps} steps")
                reward_mean = np.mean(rewards)
                f.write(f"{time.time()-start},{num_steps},{reward_mean}\n")
                num_steps = 0
                start = time.time()
                rewards = []
                obs = env.reset()
                break

import test_random_policy

    

