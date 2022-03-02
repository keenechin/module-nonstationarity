import module_env
import gym
from stable_baselines3 import PPO

hardware_env = gym.make(module_env.env_name)

policy = PPO('MlpPolicy', hardware_env, verbose=1, learning_rate=1e-3)
policy.learn(total_timesteps=10000)