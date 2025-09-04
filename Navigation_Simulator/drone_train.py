"""
Runs training on the DroneEnv.

Requires the gymnasium/envs/mujoco/mujoco_env.py to be replaced by the modified version in this repository
"""


from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium.envs.registration import register

try:
    register(
        id='Drone-v0',
        entry_point='drone_env:DroneEnv',
    )
    print("Environment registered successfully!")
except Exception as e:
    print(f"Failed to register environment: {e}")

env = gym.make("Drone-v0", render_mode="human")

# model = PPO("MlpPolicy", env, verbose=2)

model = PPO(
            "MlpPolicy",
            env,
            learning_rate=0.0007,
            n_steps=600,
            batch_size=60,
            n_epochs=20,
            gamma=0.995,
            clip_range=0.25,
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.02,
            vf_coef=0.7,
            max_grad_norm=0.5,
            target_kl=None,
            verbose=2,
            device='auto',
            _init_setup_model=True
        )
model.learn(total_timesteps=1_000_000)

# model.save("ppo_humanoid")

