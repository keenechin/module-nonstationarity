import gym
from os.path import exists
import numpy as np
from numpy import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import pickle
import pdb

from soft_fingers import get_listener_funcs

class ActionNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        self.fc1 = nn.Linear(obs_size, obs_size)
        self.fc2 = nn.Linear(obs_size, obs_size)
        self.fc3 = nn.Linear(obs_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

class DynamicsNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        self.fc1 = nn.Linear(obs_size + action_size, obs_size)
        self.fc2 = nn.Linear(obs_size, obs_size)
        self.fc3 = nn.Linear(obs_size, obs_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


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
    model = DynamicsNetwork(env)
    X = []
    y = []
    X, y = rollout(env, policy, params, N)
    model.fit(X, y)
    print("Model Trained.")
    return model, X, y


def expert_demonstration_policy(state, params):
    pass


def get_env_random_policy(env):
    def random_policy(state, params):
        return env.action_space.sample()
    return random_policy


def softmax_grad(softmax):
    s = softmax.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


def reinforce(policy, w, softmax_grad, env, model, NUM_EPISODES, LEARNING_RATE, LR_FINAL, GAMMA):
    print("Starting RL on trained dynamics model.")
    r = LR_FINAL ** (1/NUM_EPISODES)
    nA = env.action_space.shape[0]
    episode_rewards = []

    best_w = w
    best_score = 0

    for e in range(NUM_EPISODES):
        state = np.reshape(env.reset(), (1, -1))
        grads = []
        rewards = []
        score = 0
        for i in range(100):
            probs = policy(state, w)
            action = np.random.choice(list(range(nA)), p=probs[0])
            next_state = model.predict(
                np.reshape([*state[0], action], (1, -1)))
            reward = env.reward(next_state[0])
            dsoftmax = softmax_grad(probs)[action, :]
            dlog = dsoftmax / probs[0, action]
            grad = state.T.dot(dlog[None, :])
            grads.append(grad)
            rewards.append(reward)
            score += reward
            state = next_state

        for i in range(len(grads)):
            # update towards the log policy gradient times **FUTURE** reward
            w += LEARNING_RATE * \
                grads[i] * sum([r * (GAMMA ** r)
                                for t, r in enumerate(rewards[i:])])

        # Learning rate decay
        LEARNING_RATE = LEARNING_RATE * r
        # Append for logging and print
        episode_rewards.append(score)
        if score > best_score:
            best_w = w
            best_score = score
        # print(LEARNING_RATE)
        print(f"Episode: {str(e)} Score: {str(score)}")
    return episode_rewards, best_w, best_score


def get_model_contingent(learn_model, policy, env_name, env, N):
    random_collector = 'random'
    linear_collector = 'linear'
    random_data_dynamics_model = f'data/{env_name}_N{N}_{random_collector}policy'
    linear_data_dynamics_model = f'data/{env_name}_N{N}_{linear_collector}policy'
    random_data_collected = exists(random_data_dynamics_model)
    linear_data_collected = exists(linear_data_dynamics_model)
    random_policy_file = f"{random_data_dynamics_model}_trained_policy.sav"
    linear_policy_trained = exists(random_policy_file)
    linear_policy_file = f"{linear_data_dynamics_model}_trained_policy.sav"

    if linear_data_collected:
        print("Loading linear-trained dynamics model")
        w = np.random.rand(env.observation_space.shape[0], env.action_space.shape[0])
        model = pickle.load(open(f"{linear_data_dynamics_model}.sav", 'wb'))
        policy_file = linear_policy_file
    elif linear_policy_trained:
        print("Training dynamics with linear policy")
        w = pickle.load(open(random_policy_file, 'rb'))
        model, X, y = learn_model(env, policy, w, N)
        pickle.dump(model, open(f"{linear_data_dynamics_model}.sav", 'wb'))
        pickle.dump((X, y), open(f"{linear_data_dynamics_model}_data.sav", 'wb'))
        policy_file = linear_policy_file
    elif random_data_collected:
        print("Loading random-trained dynamics model")
        w = np.random.rand(env.observation_space.shape[0], env.action_space.shape[0])
        model = pickle.load(open(f"{random_data_dynamics_model}.sav", 'rb'))
        policy_file = random_policy_file
    else:
        print("Training dynamics with random policy")
        w = np.random.rand(env.observation_space.shape[0], env.action_space.shape[0])
        model, X, y = learn_model(env, policy, w, N)
        pickle.dump(model, open(f"{random_data_dynamics_model}.sav", 'wb'))
        pickle.dump((X, y), open(f"{random_data_dynamics_model}_data.sav", 'wb'))
        policy_file = random_policy_file

    return policy_file, w, model


if __name__ == "__main__":
    gym.envs.register(
        id='SoftFingerModulesEnv-v0',
        entry_point='module_env:SoftFingerModulesEnv',
        kwargs={}
    )
    # env_name = 'CartPole-v1'
    env_name = 'SoftFingerModulesEnv-v0'

    env = gym.make(env_name)
    random_policy = get_env_random_policy(env)
    np.random.seed(6)
    for i in range(1):
        N = 1000 * 5**(int(i/2))
        start = time.time()
        print(f"Collecting {N} samples.")
        policy_file, w, model = get_model_contingent(
            learn_model, random_policy, env_name, env, N)
        print(f"Time passed to collect {N} samples: {(time.time()-start)/60.0} minutes")

        # Learning Code Here
        # REINFORCE

        start = time.time()
        NUM_EPISODES = 100
        LEARNING_RATE = 0.0001
        LR_FINAL = 0.2
        GAMMA = 0.99
        episode_rewards, best_w, best_score = reinforce(
            random_policy, w, softmax_grad, env, model, NUM_EPISODES, LEARNING_RATE, LR_FINAL, GAMMA)

        # print((best_score, best_w))
        pickle.dump(best_w, open(policy_file, 'wb'))
        # import matplotlib.pyplot as plt
        # plt.plot(np.arange(NUM_EPISODES), episode_rewards)
        # plt.show()
        # Wrapup code
        print(f"Time passed for {NUM_EPISODES} RL Episodes: {(time.time()-start)/60} minutes")
        env.reset()
