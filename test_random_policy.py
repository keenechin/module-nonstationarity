import primitives_module_env
import gym
import time
import random
import numpy as np

env = gym.make(primitives_module_env.env_name)
start = time.time()
num_completions = 0
num_steps = 0
with open(f"biased_random_task_completion_{primitives_module_env.env_name}.txt", 'w') as f:
    rewards = []
    while num_completions<10 and (time.time()-start)<3600:
        num_steps += 1
        action = random.choice(range(env.action_size//2))
        state, reward, done, _ = env.step(action)
        rewards.append(reward)
        if done:
            reward_mean = np.mean(rewards)
            f.write(f"{time.time()-start},{num_steps},{reward_mean}\n")
            num_steps = 0
            num_completions += 1
            start = time.time()
            rewards = []
            env.reset()

