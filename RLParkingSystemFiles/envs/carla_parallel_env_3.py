from envs.carla_parallel_env import CarlaParallelParkingEnv
import numpy as np
import gymnasium as gym

from sensors.data_fusion import lidar_to_occupancy_grid


class CarlaParallelParkingEnhancedObsEnv(CarlaParallelParkingEnv):
    def __init__(self, config=None, grid_size=(60, 60)):
        super().__init__(config=config, grid_size=grid_size)

        # New observation length = yaw + speed + ogm + 6 distances + gear
        obs_len = 2 + self.grid_size_width * self.grid_size_height + 6 + 1
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_len,),
            dtype=np.float32
        )

    def _get_observation(self):
        lidar_pts = self.sensor_manager.get_lidar_data()
        ogm_1d = lidar_to_occupancy_grid(lidar_pts, self.grid_size_width, self.grid_size_height, 0.2)
        tf = self.ego_vehicle.get_transform()
        velocity = self.ego_vehicle.get_velocity()

        yaw = tf.rotation.yaw
        speed = np.sqrt(velocity.x ** 2 + velocity.y ** 2)
        distances = self._compute_vehicle_distances()

        obs = np.zeros(2 + len(ogm_1d) + 6 + 1, dtype=np.float32)
        obs[0] = yaw
        obs[1] = speed
        obs[2:2+len(ogm_1d)] = ogm_1d
        obs[2+len(ogm_1d):2+len(ogm_1d)+6] = [
            distances['rear_center_dist'],
            distances['rear_left_dist'],
            distances['rear_right_dist'],
            distances['front_center_dist'],
            distances['front_left_dist'],
            distances['front_right_dist']
        ]
        obs[-1] = float(self.gear)

        return obs
