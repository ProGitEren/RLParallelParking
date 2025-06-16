import time
import gym
import numpy as np
from stable_baselines3 import PPO
from rl.trainer import train_ppo_carla
from envs.carla_parallel_env import CarlaParallelParkingEnv
from stable_baselines3.common.vec_env import DummyVecEnv

def evaluate_model(model_path="logs/models/ppo_parallel_parking.zip",episodes=10):
    env = DummyVecEnv([lambda: CarlaParallelParkingEnv(config={"max_steps":400})])
    model = PPO.load(model_path, env=env)

    success_count = 0

    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _states = model.predict(obs,deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward

        print(f"Episode {ep} reward: {total_reward:.2f}")

        if env.envs[0].has_collided is False and env.envs[0].successful is True:
            success_count += 1

    print(f"Success Rate: {success_count/episodes*100:.2f}%")

if __name__ == "__main__":
    evaluate_model()