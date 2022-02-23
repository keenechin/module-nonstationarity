from queue import Empty
from soft_fingers import ExpertActionStream, SoftFingerModules
import numpy as np
import module_env
import gym



def generate_expert_traj(env, n_episodes=3):
    actions = []
    observations = []
    rewards = []
    episode_returns = np.zeros((n_episodes,))
    episode_starts = []

    ep_idx = 0
    obs = env.reset()
    episode_starts.append(True)
    reward_sum = 0.0



    
    with ExpertActionStream(manipulator=env, target_theta=0) as action_listener:
        obj, process, action_channel, state_channel = action_listener

        while process.is_alive() and ep_idx < n_episodes:
            try:
                command = action_channel.get(False)
                try:
                    action = obj.parse(command)
                    obs, reward, done, _ = env.step(action)
                    while not state_channel.empty():
                        state_channel.get()
                    state_channel.put(env.object_pos)
                    observations.append(obs)
                    actions.append(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    reward_sum += reward

                    if done:
                        env.reset()
                        episode_returns[ep_idx] = reward_sum
                        reward_sum = 0.0
                        ep_idx += 1


                except TypeError:
                    pass
            except Empty:
                pass
            
        env.reset()
        if ep_idx < n_episodes:
            raise RuntimeError(f"Expert stream closed after {ep_idx} episodes, expected {n_episodes}.")

        actions = np.array(actions)
        observations = np.array(observations)
        rewards = np.array(rewards)
        episode_returns = np.array(episode_returns)
        episode_starts = np.array(episode_starts)
        numpy_dict = {
            'actions': actions,
            'obs': observations,
            'rewards': rewards,
            'episode_returns': episode_returns,
            'episode_starts': episode_starts
        }  # type: Dict[str, np.ndarray]

        for key, val in numpy_dict.items():
            print(key, val.shape)

        prompt = "Name these trajectories:"
        save_path = f"{n_episodes}e_{input(prompt)}"
        if save_path is not None:
            np.savez(save_path, **numpy_dict)

if __name__ == "__main__":
    env = gym.make(module_env.env_name)
    generate_expert_traj(env)