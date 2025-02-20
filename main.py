import gymnasium as gym

from rainbow_dqn.dqn_agent import DQNAgent

env = gym.make("LunarLander-v3", max_episode_steps=200, render_mode="rgb_array")
seed = 42
num_frames = 10000
memory_size = 10000
batch_size = 128
target_update = 100

import time

start_time = time.time()

agent = DQNAgent(env, memory_size, batch_size, target_update, seed)
agent.train(num_frames, plotting_interval=20000)
agent.test()

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")