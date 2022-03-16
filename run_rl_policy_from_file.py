import module_env
import gym
from stable_baselines3 import PPO
import tkinter as tk

tk.Tk().withdraw()
env =  gym.make(module_env.env_name)
# policy_fname = tk.filedialog.askopenfilename()
policy_fname = "policies/PPO_policy_N32768.zip"
policy = PPO.load(policy_fname)
obs = env.reset()
steps_to_complete = 0
print(f"Starting at {obs}")
while True:
    steps_to_complete += 1
    action = policy.predict(obs)[0]
    # action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    if done:
        print(f"Finished in {steps_to_complete} steps")
        break
    timeout = 1000
    if steps_to_complete >= timeout:
        print(f"Failed to complete within {timeout} steps.")
        break

