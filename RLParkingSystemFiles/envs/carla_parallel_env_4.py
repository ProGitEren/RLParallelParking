from pandas.io.formats.info import frame_examples_sub

from envs.carla_parallel_env import CarlaParallelParkingEnv
import gymnasium as gym
import numpy as np
import carla
import os
import cv2

from sensors.data_fusion import lidar_to_occupancy_grid

# Define text parameters

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.5
thickness = 3
color = (0, 0, 0)  # white text

def put_episode_text(frame, text):
    frame = frame.copy()  # make it writable
    # Get frame width and height
    (h, w) = frame.shape[:2]

    # Get text size
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    # Calculate coordinates: top-center and a bit down from the top
    x = (w - text_width) // 2
    y = int(0.08 * h)  # 8% down from the top; tweak this value as needed
    # Put text on frame
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    return frame


class CarlaParallelParkingHybridEnv(CarlaParallelParkingEnv):
    def __init__(self, config=None, grid_size=(60, 60)):
        super().__init__(config=config, grid_size=grid_size)

        self.record_video = True  # Optional flag to control recording
        self.video_path = "logs/hybrid_recording.mp4"
        self.concat_video_path = "logs/hybrid_recording_concat.mp4"
        self.video_fps = 20
        self.video_writer = None
        self.concat_video_writer = None


        # Reduced and optimized discrete action space
        self.steer_values = [-0.5, -0.4, -0.3, -0.1, 0.0, 0.1, 0.3, 0.4, 0.5]
        self.throttle_values = [-0.5, -0.2, 0.0, 0.2, 0.3]
        self.gear_values = [0, 1]  # 0 = forward, 1 = reverse

        self.discrete_actions = [
            (s, t, g) for s in self.steer_values
                      for t in self.throttle_values
                      for g in self.gear_values
        ]

        # New observation length = yaw + speed + ogm + 6 distances + gear
        obs_len = 2 + self.grid_size_width * self.grid_size_height + 6 + 1
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_len,),
            dtype=np.float32
        )

        self.action_space = gym.spaces.Discrete(len(self.discrete_actions))
        self.log_file = open("logs/hybrid_discrete_log.txt", "w")

    def step(self, action_idx):
        self.current_step += 1
        action = self.discrete_actions[action_idx]
        self._apply_control(action)
        self.world_mgr.tick()

        self.world_mgr.follow_ego_vehicle()

        frame = self.sensor_manager.get_bev_camera_frame()  # or get_camera_frame()

        obs = self._get_observation()
        gt_yaw = self._ground_truth_yaw()
        distances = self._compute_vehicle_distances()
        reward, done = self._compute_reward_and_done(distances, gt_yaw)

        self._update_phase_logic(distances, gt_yaw, self.parking_side)
        self._log_step((action[0], action[1], action[2]), obs, distances, gt_yaw, reward, done)

        terminated = done
        truncated = self.current_step >= self.max_steps and not terminated

        if (self.record_video) and (self.video_writer is not None) and (self.concat_video_writer is not None) and (
                frame is not None):
            # Concat visualization
            ogm_vis = self._get_ogm_visual(frame)
            concat_frame = cv2.hconcat([frame, ogm_vis])

            # Add episode text
            text = f"Episode {self.episode}"

            frame = put_episode_text(frame, text)
            concat_frame = put_episode_text(concat_frame, text)

            self.video_writer.write(frame)
            self.concat_video_writer.write(concat_frame)

        return obs, reward, terminated, truncated, {}

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


