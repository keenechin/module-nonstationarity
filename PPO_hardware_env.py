from locale import normalize
import primitives_module_env
import gym
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
import time

hardware_env = gym.make(primitives_module_env.env_name)
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
    for i in range(14, 17):
        hardware_env.reset()
        learn_steps = 2 ** i
        policy = PPO('MlpPolicy', hardware_env, learning_rate=1e-3, 
                    verbose=2, n_steps=update_steps, batch_size=update_steps, 
                    tensorboard_log=f"policies/log_{learn_steps}")
        print(f"Training PPO with {learn_steps} steps.")
        policy.learn(total_timesteps=learn_steps, log_interval=10)
        ppo_file = f"policies/PPO_policy_N{learn_steps}"
        if save:
            policy.save(ppo_file)
        wait = np.min((2 ** (i-5), 1800))
        print(f"Cooling down for {wait/60} minutes.")
        time.sleep(wait)
    policy = PPO.load(ppo_file, print_system_info=True)

obs = hardware_env.reset()
for i in range(13):
    action = policy.predict(obs)[0]
    obs, reward, done, _ = hardware_env.step(action)
    if done:
        break
    

