from locale import normalize
import discrete_module_env
import gym
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
import time

class Logger:
    def __init__(self, fname):
        self.fname = fname
        self.file = open(fname, 'w')
        self.step = 0
        
    def print_func(self, locals, globals):
        self.step = self.step+1
        self.file.write(f"{self.step}, {locals['rewards'][0]}\n")
    
    def __del__(self):
        self.file.close()
hardware_env = gym.make(discrete_module_env.env_name)
save = True
ddpg = False
if ddpg:
    n_actions = hardware_env.action_space.shape[0]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    policy = DDPG('MlpPolicy', hardware_env, verbose=2, learning_rate=1e-3 )
    policy.learning_starts = 100
    policy.learn(total_timesteps=100, log_interval=10)
    time.sleep(100)
elif save:
    update_steps = 64
    for i in range(6, 18):
        hardware_env.reset()
        learn_steps = 2 ** i
        policy = PPO('MlpPolicy', hardware_env, learning_rate=1e-3, 
                    verbose=2, n_steps=update_steps, batch_size=update_steps, 
                    tensorboard_log=f"policies/log_{learn_steps}")
        print(f"Training PPO with {learn_steps} steps.")
        policy.learn(callback=Logger(f"policies/log_{learn_steps}.txt").print_func, total_timesteps=learn_steps, log_interval=1)
        ppo_file = f"policies/PPO_policy_N{learn_steps}"
        if save:
            policy.save(ppo_file)
        wait = np.min((2 ** (i-4), 1800))
        print(f"Cooling down for {wait/60} minutes.")
        time.sleep(wait)
    policy = PPO.load(ppo_file, print_system_info=True)

# obs = hardware_env.reset()
# for i in range(13):
#     action = policy.predict(obs)[0]
#     obs, reward, done, _ = hardware_env.step(action)
#     if done:
#         break
    

