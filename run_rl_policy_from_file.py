import primitives_module_env
import gym
from stable_baselines3 import PPO
import tkinter as tk
import time
import numpy as np

tk.Tk().withdraw()
env =  gym.make(primitives_module_env.env_name)
policy_fname = tk.filedialog.askopenfilename()
policy_size = policy_fname.split("_")[-1][:-4]
policy = PPO.load(policy_fname)
obs = env.reset()
num_steps = 0
# print(f"Starting at {obs}")
start = time.time()
import os
logdir = "policy_eval_logs"
files = [f for f in os.listdir(logdir) if os.path.isfile(os.path.join(logdir,f))]
lognum = 1
for file in files:
    file_lognum = int(file.split("_")[1])
    if file_lognum >= lognum:
        lognum = file_lognum + 1


evaltype = input("What kind of finger for the policy? \nType original, silicone, thicker, or longer: ")
trained = input("What kind of finger is being evaluated? \nType original, silicone, thicker, or longer: ")
policy_eval_fname = f"{logdir}/log_{lognum}_{policy_size}_size_{trained}_trained_{evaltype}_eval.txt"
print(f"Saving to {policy_eval_fname}")
with open(policy_eval_fname, 'w') as f:
    rewards = []
    for i in range(int(input("Enter num trials:"))):
        while True:
            num_steps += 1
            action = policy.predict(obs)[0]
            print(action)
            # action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            rewards.append(reward)
            timeout = (time.time()-start)>60
            if done or timeout:
                print(f"Done: {done}, Timed out:{timeout}")
                print(f"Finished in {num_steps} steps")
                reward_mean = np.mean(rewards)
                f.write(f"{time.time()-start},{num_steps},{reward_mean}\n")
                num_steps = 0
                start = time.time()
                rewards = []
                obs = env.reset()
                break


    

