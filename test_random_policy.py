import discrete_module_env
import gym
import time
import random

env = gym.make(discrete_module_env.env_name)
start = time.time()
num_completions = 0
num_steps = 0
with open(f"random_task_completion_{discrete_module_env.env_name}.txt", 'w') as f:
    while num_completions<10 and (time.time()-start)<3600:
        num_steps += 1
        state, reward, done, _ = env.step(random.choice(range(env.action_size)))
        if done:
            f.write(f"{time.time()},{num_steps}\n")
            num_steps = 0
            num_completions += 1
            env.reset()

