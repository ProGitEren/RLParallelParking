import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.carla_parallel_env import CarlaParallelParkingEnv
from envs.carla_parallel_env_4 import CarlaParallelParkingHybridEnv
from rl.policy import CustomParkingPolicyDiscrete, CustomParkingPolicyContinuous, HybridParkingPolicyDiscrete
from  rl.visualize_callback import FeatureLoggingCallback


def train_ppo_carla(config=None, total_timesteps=300_000,load_from=None,save_path="logs/models/ppo_parallel_parking"):

    def make_env():
        return CarlaParallelParkingEnv(config=config)

    # stable-baselines3 often requires a vec env
    # This can be later on used tobe parallel CARLA instances on different ports or machines for faster training and diverse samples per update
    env = DummyVecEnv([make_env])

    if load_from:
        model = PPO.load(load_from, env=env)
    else:
        # Possibly configure policy_kwargs
        model = PPO(
            policy=CustomParkingPolicyContinuous,
            env=env,
            verbose = 1,
            n_steps = 1024,
            batch_size = 64,
            learning_rate = 3e-4,
            gamma = 0.99,
            n_epochs=5,
            tensorboard_log="./logs/tensorboard/"

        )

    model.learn(total_timesteps=total_timesteps)

    model.save(save_path)
    return model

def train_ppo_carla_discrete(config=None, total_timesteps=300_000,load_from=None,save_path="logs/models/ppo_parallel_parking"):
    def make_env():
        return CarlaParallelParkingHybridEnv(config=config)

    env = DummyVecEnv([make_env])

    if load_from:
        model = PPO.load(load_from, env=env)
    else:
        model = PPO(
            policy=HybridParkingPolicyDiscrete,
            env=env,
            verbose=1,
            n_steps=1024,
            batch_size=64,
            learning_rate=3e-4,
            gamma=0.99,
            n_epochs=5,
            tensorboard_log="./logs/tensorboard/"
        )

    callback = FeatureLoggingCallback(log_dir="logs/features")

    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(save_path)
    env.close()
    return model