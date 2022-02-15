from queue import Empty
from soft_fingers import ExpertActionStream, SoftFingerModules
import numpy as np
import module_env
import gym



def generate_expert_traj(env, n_timesteps=0, n_episodes=100):
    actions = []
    observations = []
    rewards = []
    episode_returns = np.zeros((n_episodes,))
    episode_starts = []

    ep_idx = 0
    obs = env.reset()
    episode_starts.append(True)
    reward_sum = 0.0
    idx = 0


    
    with ExpertActionStream(manipulator=env, target_theta=0) as action_listener:
        obj, process, action_channel, state_channel = action_listener
        while process.is_alive() and ep_idx < n_episodes:
            try:
                command = action_channel.get(False)
                try:
                    action = obj.parse(command)
                    env.step(action)
                    while not state_channel.empty():
                        state_channel.get()
                    state_channel.put(env.object_pos)
                except TypeError:
                    pass
            except Empty:
                pass
            

            
        
        if ep_idx < n_episodes:
            raise RuntimeError(f"Expert stream closed after {ep_idx} episodes, expected {n_episodes}.")
        env.reset()

if __name__ == "__main__":
    env = gym.make(module_env.env_name)
    generate_expert_traj(env)