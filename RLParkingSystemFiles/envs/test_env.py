import time
from envs.carla_parallel_env_4 import CarlaParallelParkingHybridEnv
from envs.carla_parallel_env import CarlaParallelParkingEnv

if __name__ == "__main__":
    env = CarlaParallelParkingHybridEnv(config={"max_steps": 200}, grid_size= (60, 40))
    obs = env.reset()
    print("Environment Initialized and Reset!")
    step = 0
    done = False
    trunc = False
    gear = 0
    for _ in range(2):

        while not done and not trunc:
            step += 1
            action_index = 49
            if step == 10:
                #gear = 1
                action_index = 40
            if step == 20:
                #gear = 0
                action_index = 49

            #action = env.action_space.sample()  # Take a random action
            action = (0.3,0.2,gear)
            obs, reward, done, trunc, info = env.step(action_index)
            env.render()    # Optional: show LiDAR occupancy grid
            time.sleep(0.05)  # Slow down to human-readable speed

        done = False
        trunc = False
        env.reset()
        print("Environment Reset!")

    env.close()
    print("Env is closed")
