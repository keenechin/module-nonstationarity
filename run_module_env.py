import module_env
import gym
env = gym.make(module_env.env_name)
env.reset()
for i in range(100):
    action = env.action_space.sample()
    print(action)
    env.step(action)
    input("Press any key to continue.")
    
env.reset()