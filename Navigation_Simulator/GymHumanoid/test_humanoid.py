import gymnasium as gym

env = gym.make("Humanoid-v4", render_mode="human")  # Use "human" for GUI rendering
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
