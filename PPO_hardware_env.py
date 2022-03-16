from locale import normalize
import module_env
import gym
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
import time

hardware_env = gym.make(module_env.env_name)
update_steps = 64
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
# policy = DDPG('MlpPolicy', hardware_env, verbose=2, learning_rate=1e-3 )
# policy.learning_starts = 15
# policy.learn(total_timesteps=1000, log_interval=10)
# time.sleep(100)
for i in range(6, 16):
    hardware_env.reset()
    learn_steps = 2 ** i
    policy = PPO('MlpPolicy', hardware_env, learning_rate=1e-3, verbose=2, n_steps=update_steps, batch_size=update_steps)
    print(f"Training PPO with {learn_steps} steps.")
    policy.learn(total_timesteps=learn_steps, log_interval=10)
    ppo_file = f"policies/PPO_policy_N{learn_steps}"
    # policy.save(ppo_file)
    wait = np.min((2 ** (i-4), 1800))
    print(f"Cooling down for {wait/60} minutes.")
    time.sleep(wait)
learned = PPO.load(ppo_file, print_system_info=True)

obs = hardware_env.reset()
for i in range(13):
    action = learned.predict(obs)[0]
    obs, reward, done, _ = hardware_env.step(action)
    if done:
        break
    

