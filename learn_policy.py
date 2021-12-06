import gym
from os.path import exists
import numpy as np
from numpy import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import pickle
import pdb
from soft_fingers import ExpertActionStream

def rollout(env, policy, params, num_steps):
    X = []
    y = []
    state = env.reset()
    nA = env.action_space.shape[0]

    for i in range(num_steps):
        action = policy(state, params)
        next_state, reward, _, _ = env.step(action)
        X.append([*state, action])
        y.append([*next_state])
        
        state = next_state
    return X, y


def learn_model(env, policy, params, N):
    input_size = env.observation_space.shape[0] + env.action_space.shape[0]
    output_size = env.observation_space.shape[0]
    hidden_size = (input_size + output_size)//2
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size)
    )
    X = []
    y = []
    X, y = rollout(env, policy, params, N)
    model.fit(X, y)
    print("Model Trained.")
    return model, X, y




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
    print(f'Using {device} device')
    np.random.seed(6)