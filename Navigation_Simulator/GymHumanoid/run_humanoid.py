import argparse
from stable_baselines3 import PPO
import gymnasium as gym

def main(model_path):
    # Create environment with GUI
    env = gym.make("Humanoid-v4", render_mode="depth_array")

    # Load model
    model = PPO.load(model_path)

    # Run the trained policy
    obs, info = env.reset()
    for _ in range(10000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the saved PPO model (e.g., ppo_humanoid.zip)")
    args = parser.parse_args()

    main(args.model)
