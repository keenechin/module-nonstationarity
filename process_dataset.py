import pickle
from module_env import SoftFingerModulesEnv
import numpy as np


def get_fname(num_steps):
    return f"data/dataset_N{num_steps}.pickle"



def accumulate_data(env, num_points, num_files=7, visualize=False):
    dataset = []
    for i in range(num_files):
        num_steps = 50 * 2 **(i+1)
        with open(get_fname(num_steps), 'rb') as file:
            filedata = pickle.load(file)
            if len(dataset) == 0:
                dataset = filedata
            else:
                dataset = np.vstack((dataset, filedata))
    
            
            if visualize:
                import matplotlib.pyplot as plt
                plt.scatter(dataset[:,12], dataset[:,13])
                plt.title(f"Positions of valve over time for {num_steps} steps")
                plt.axis('square')
                plt.show()

        
    dObs = env.observation_space.shape[0]
    dAct = env.action_space.shape[0]
    x = dataset[:num_points, 0:dObs+dAct]
    y = dataset[:num_points, dObs+dAct:2*dObs+dAct]
    r = dataset[:num_points, -1]
    return x, y,r

if __name__ == "__main__":
    env = SoftFingerModulesEnv()
    x,y,r = accumulate_data(env, 51200, 10, visualize=True)