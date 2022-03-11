from locale import normalize
import module_env
import gym
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

hardware_env = gym.make(module_env.env_name)
n_actions = hardware_env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
policy = DDPG('MlpPolicy', hardware_env, action_noise=action_noise, verbose=2, learning_rate=1e-3 )
policy.learning_starts = 10
# policy.policy.set_training_mode(False)
#policy = PPO('MlpPolicy', hardware_env, learning_rate=1e-2, verbose=2)
policy.learn(total_timesteps=10000, log_interval=10)
