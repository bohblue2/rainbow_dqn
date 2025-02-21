import gymnasium as gym

from rl_agent_zoo.rainbow import RainbowDqnAgent 

env = gym.make("LunarLander-v3", max_episode_steps=200, render_mode="rgb_array")
seed = 42
num_frames = 50000
memory_size = 20000
batch_size = 128
target_update = 500

import time

start_time = time.time()

agent = RainbowDqnAgent(env, memory_size, batch_size, target_update, seed)
agent.train(num_frames, plotting_interval=20000)
agent.test()

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")