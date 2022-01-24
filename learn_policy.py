import gym
from queue import Empty
import numpy as np
import torch
import torch.nn as nn
import time
import pickle
import tkinter as tk
from tkinter import filedialog
from soft_fingers import ExpertActionStream
from enum import Enum, auto
from process_dataset import accumulate_data
from learned_dynamics_env import LearnedDynamicsEnv


class DynamicsModel():
    def __init__(self, input_size, output_size):
        hidden_size = (input_size + output_size)//2
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

        self.loss_fn = nn.MSELoss()

    def fit(self, X, y, learning_rate = 3e-3):
        optimizer = torch.optim.SGD(self.network.parameters(), lr=learning_rate)
        batch_size = 5
        num_batches = len(X)//batch_size
        for batch in range(num_batches):
            batch_X = X[batch_size * batch : batch_size * (batch+1) + 1]
            batch_y = y[batch_size * batch : batch_size * (batch+1) + 1]

            pred = self.network(batch_X)
            loss = self.loss_fn(pred, batch_y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(batch_X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{len(X):>5d}]")


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
        if (num_steps % 1000) == 0:
            print(f"Completed {i} steps, cooling down for 30s.")
            time.sleep(30)
    return torch.from_numpy(X).type(torch.FloatTensor), torch.from_numpy(y).type(torch.FloatTensor), rewards

def get_model(env):
    input_size = env.observation_space.shape[0] + env.action_space.shape[0]
    output_size = env.observation_space.shape[0]
    model = DynamicsModel(input_size, output_size)
    return model

def learn_model(model, X, y, rewards):
    model.fit(X,y)
    print("Model Trained.")
    return model, X, y, rewards


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

class DynamicsSource(Enum):
    EXPERT_POLICY = auto()
    RANDOM_POLICY = auto()
    SAVED_DATA = auto()
    TRAINED_MODEL = auto()


def test_model(rollout, env, random_policy, model, num_steps):
    X_traj, Y_traj, rewards = rollout(env, random_policy, None, num_steps)

            # Test
    test_loss = 0
    with torch.no_grad():
        for X,Y  in zip(X_traj, Y_traj):
            pred = model.predict(X)
            test_loss += model.loss_fn(pred, Y)
    test_loss = test_loss / num_steps
    print(f"Loss: {test_loss}")

def import_model():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    import time
    time.sleep(4)
    return model

def get_dynamics_model(env, mode = DynamicsSource.SAVED_DATA):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device for torch.')
    np.random.seed(7)
    if mode == DynamicsSource.TRAINED_MODEL:
        return import_model()

    target_theta = env.nominal_theta
    random_policy = create_random_policy(env)

    num_files = 1
    for i in range(num_files):
        num_steps = 50 * 2 **(i+1)
        model = get_model(env)
        
        if mode == DynamicsSource.SAVED_DATA or DynamicsSource.TRAINED_MODEL:
            save_mode = False
        else:
            save_mode = True
            print(f"Collecting {num_steps} sample dataset.")


        if mode == DynamicsSource.EXPERT_POLICY:
            with ExpertActionStream(env.hardware, target_theta) as action_listener:
                process, action_channel, state_channel = action_listener
                expert_policy = create_expert_policy(env, action_channel, state_channel)
                x, y, rewards = rollout(env, expert_policy, None, num_steps)
                model, x, y, rewards = learn_model(model, x, y, rewards)
                # test_model(rollout, env, expert_policy, model, 5)

        if mode == DynamicsSource.RANDOM_POLICY:
            x, y, rewards = rollout(env, random_policy, None, num_steps)
            model, x, y, rewards = learn_model(model, x, y, rewards)
            # test_model(rollout, env, random_policy, model, 5)

        if mode == DynamicsSource.SAVED_DATA:
            num_training = 62600
            x, y, rewards = accumulate_data(env, num_points=num_training)
            x = torch.from_numpy(x).type(torch.FloatTensor)
            y = torch.from_numpy(y).type(torch.FloatTensor)
            rewards = torch.from_numpy(rewards).type(torch.FloatTensor)

            model, x, y, rewards = learn_model(model, x, y, rewards)
            with open(f"trained_forward_model_N{num_training}.pickle", 'wb') as f:
                pickle.dump(model, f)
            # test_model(rollout, env, random_policy, model, 40)

        if save_mode:
            dataset = np.hstack((x, y , torch.unsqueeze(rewards, 1)))
            with open(f"data/dataset_N{num_steps}.pickle", 'wb') as f:
                pickle.dump(dataset, f)
            print(f"{num_steps} sample dataset collected, cooling down.")
            env.reset()
            time.sleep(60)

        time.sleep(1)
        env.reset()
        return model

if __name__ == "__main__":
    gym.envs.register(
        id='SoftFingerModulesEnv-v0',
        entry_point='module_env:SoftFingerModulesEnv',
        kwargs={}
    )
    hardware_env_name = 'SoftFingerModulesEnv-v0'
    hardware_env = gym.make(hardware_env_name)
    dynamics = get_dynamics_model(hardware_env, mode=DynamicsSource.TRAINED_MODEL)

    gym.envs.register(
        id='ModulesSimulationEnv-v0',
        entry_point='learned_dynamics_env:LearnedDynamicsEnv',
        kwargs={'hardware_env':hardware_env, 'dynamics':dynamics}
    )
    software_env_name = 'ModulesSimulationEnv-v0'
    software_env = gym.make(software_env_name)
    
    


