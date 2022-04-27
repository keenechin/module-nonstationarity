from queue import Empty
from manual_controller import ExpertActionStream
import numpy as np
import discrete_module_env
import module_env
import gym



def generate_expert_traj(env, n_episodes=10, expected_ep_ln=50):
    actions = [None] * n_episodes * expected_ep_ln
    observations = [None] * n_episodes * expected_ep_ln
    rewards = [None] * n_episodes * expected_ep_ln
    episode_returns = np.zeros((n_episodes,))
    episode_starts = [None] * n_episodes * expected_ep_ln

    total_idx = 0
    ep_idx = 0
    obs = env.reset()
    episode_starts.append(True)
    reward_sum = 0.0



    
    with ExpertActionStream(manipulator=env, target_theta=0) as action_listener:
        obj, process, action_channel, state_channel = action_listener

        while process.is_alive():
            try:
                command = action_channel.get(False)
                try:
                    action = env.interpret(command)
                    obs, reward, done, _ = env.step(action)
                    while not state_channel.empty():
                        state_channel.get()
                    state_channel.put(env.object_pos)


                    try:
                        observations[total_idx] = obs                    
                        actions[total_idx] = action
                        rewards[total_idx] = reward
                        episode_starts[total_idx] = done
                    except IndexError:
                        observations.append(obs)
                        actions.append(action)
                        rewards.append(reward)
                        episode_starts.append(done)

                    total_idx += 1

                    reward_sum += reward

                    if done:
                        env.reset()
                        episode_returns[ep_idx] = reward_sum
                        reward_sum = 0.0
                        ep_idx += 1
                        if ep_idx >= n_episodes:
                            break


                except TypeError:
                    pass
            except Empty:
                pass
            
        env.reset()
        if ep_idx < n_episodes:
            raise RuntimeError(f"Expert stream closed after {ep_idx} episodes, expected {n_episodes}.")

        while actions[-1] is None:
            actions.pop()
            observations.pop()
            rewards.pop()
            episode_starts.pop()

        actions = np.array(actions)
        observations = np.array(observations)
        rewards = np.array(rewards)
        episode_starts = np.array(episode_starts)
        episode_returns = np.array(episode_returns)
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
        save_path = f"trajectories/{n_episodes}e_{input(prompt)}"
        if save_path is not None:
            np.savez(save_path, **numpy_dict)

if __name__ == "__main__":
    env = gym.make(discrete_module_env.env_name)
    generate_expert_traj(env, n_episodes=20)