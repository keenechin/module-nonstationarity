import csv
import numpy as np
import matplotlib.pyplot as plt


powers = range(14,18)
for power in powers:
    average_rewards = []
    num_steps = []
    with open(f"policies/log_{2**power}.txt",'r') as f:
        reader = csv.reader(f, delimiter=',')
        rewards = []
        time_step = 0
        for row in reader:
            index = int(row[0])
            time_step += 1
            reward = float(row[1])
            rewards.append(reward)
            if reward > 0:
                average_rewards.append(np.mean(rewards))
                num_steps.append(time_step)
                rewards = [] 
                time_step = 0 
        plt.figure()
        plt.title(f"Episode Steps to Complete, N=2^{power}")
        plt.scatter(list(range(len(num_steps))), num_steps, s=2, c='r')
        plt.figure()
        plt.title(f"Average Episode Reward, N=2^{power}")
        plt.scatter(list(range(len(num_steps))), average_rewards, s=2, c='g')
        plt.show()