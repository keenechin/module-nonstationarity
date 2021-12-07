from multiprocessing import process
import gym
from queue import Empty
from os.path import exists
import numpy as np
from numpy import random
from numpy.core.numeric import roll
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import pickle
import pdb
from soft_fingers import ExpertActionStream


class DynamicsModel():
    def __init__(self, input_size, output_size):
        hidden_size = (input_size + output_size)//2
        self.network = torch.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def fit(self, X, y):
        pass

    def predict(self, X):
        return self.network(X)

def rollout(env, policy, params, num_steps):
    X = np.empty((num_steps, env.observation_size + env.action_size ))
    y = np.empty((num_steps, env.observation_size))
    rewards = np.empty((num_steps, 1))
    state = env.reset()

    for i in range(num_steps):
        action = policy(state, params)
        next_state, reward, _, _ = env.step(action)
        X[i,:] = [*state, *action]
        y[i,:] = [*next_state]
        rewards[i] = reward
        state = next_state
    return X, y, rewards

def learn_model(env, policy, params, N):
    input_size = env.observation_space.shape[0] + env.action_space.shape[0]
    output_size = env.observation_space.shape[0]
    model = DynamicsModel(input_size, output_size)
    X = []
    y = []
    X, y, rewards = rollout(env, policy, params, N)
    model.fit(X,y)
    print("Model Trained.")
    return model, X, y


def create_random_policy(env):
    def random_policy(state, params):
        action = env.action_space.sample()
        return action
    return random_policy


def create_expert_policy(env, expert_listener, state_channel):
    def expert_policy(state, params):
        while True:
            try:
                command = expert_listener.get()
                try: 
                    action = env.hardware.parse(command)
                    break
                except TypeError:
                    pass
            except Empty:
                pass
        while not state_channel.empty():
            state_channel.get()
        state_channel.put(state[-1])
        return action
    return expert_policy

if __name__ == "__main__":
    gym.envs.register(
        id='SoftFingerModulesEnv-v0',
        entry_point='module_env:SoftFingerModulesEnv',
        kwargs={}
    )
    # env_name = 'CartPole-v1'
    env_name = 'SoftFingerModulesEnv-v0'

    env = gym.make(env_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device for torch.')
    np.random.seed(6)
    num_steps = 120
    target_theta = env.nominal_theta

    with ExpertActionStream(env.hardware, target_theta) as action_listener:
        process, action_channel, state_channel = action_listener
        X,y,rewards = rollout(env, create_expert_policy(env, action_channel, state_channel), None, num_steps)
    print(f"X: {X}\ny: {y}\nRewards: {rewards} ")