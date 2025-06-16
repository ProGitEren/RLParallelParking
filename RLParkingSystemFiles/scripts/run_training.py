from torch.cuda import device

from rl.trainer import train_ppo_carla, train_ppo_carla_discrete
import matplotlib.pyplot as plt
import os
import torch


def train_cont_env1():
    NUM_ITERATIONS = 5
    total_timesteps = 10300

    model = None
    model_path = None

    config = {
        "town": "Town03",
        "max_steps": 350,
        "total_timesteps": total_timesteps
    }

    for i in range(NUM_ITERATIONS):
        save_path = f"logs/models_2/ppo_iter_{i}.zip"
        load_from = model_path if i > 0 else None

        print(f"\n🔁 Starting PPO Iteration {i + 1}/{NUM_ITERATIONS}")
        print(f"    Loading from: {load_from or 'scratch'}")
        print(f"    Saving to: {save_path}")

        model = train_ppo_carla(
            config=config,
            total_timesteps=total_timesteps,
            load_from=load_from,
            save_path=save_path
        )

        model_path = save_path  # for next iteration to load from

        inp = input("Do you want to stop ??")
        inp_low = inp.lower()
        if inp_low == "y":
            print("Early stopping !!")
            break

    print(f"\n✅ Training completed. Final model saved at: {model_path}")

def train_disc_env4(num_iterations= 5, total_timesteps=10240):

    model = None
    model_path = None

    config = {
        "town": "Town03",
        "max_steps": 350,
        "total_timesteps": total_timesteps
    }

    for i in range(num_iterations):
        save_path = f"logs/models_discrete_1/ppo_iter_{i}.zip"
        load_from = model_path if i > 0 else None

        print(f"\n🔁 Starting PPO Iteration {i + 1}/{num_iterations}")
        print(f"    Loading from: {load_from or 'scratch'}")
        print(f"    Saving to: {save_path}")

        model = train_ppo_carla_discrete(
            config=config,
            total_timesteps=total_timesteps,
            load_from=load_from,
            save_path=save_path
        )

        model_path = save_path  # for next iteration to load from

        inp = input("Do you want to stop ??")
        inp_low = inp.lower()
        if inp_low == "y":
            print("Early stopping !!")
            break

    print(f"\n✅ Training completed. Final model saved at: {model_path}")

if __name__ == "__main__":

    print("!!! START EXECUTION !!!")
    train_disc_env4(num_iterations=5,total_timesteps=10240)
    print("!!! END EXECUTION !!!")

