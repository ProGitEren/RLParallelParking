import cv2
import gymnasium as gym
import numpy as np
import carla
import os
import math
import random
from collections import deque
import time

from envs.carla_world import CarlaWorldManager
from sensors.sensor_manager import SensorManager
from sensors.data_fusion import lidar_to_occupancy_grid
from graphics.viz_occupancy import show_occupancy_grid

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


class CarlaParallelParkingEnv(gym.Env):

    def __init__(self, config=None, grid_size=(60, 60)):
        super().__init__()
        self.config = config or {}
        self.max_steps = self.config.get('max_steps', 300)
        self.grid_size_width = grid_size[1]
        self.grid_size_height = grid_size[0]

        self.episode = 0

        self.record_video = True  # Optional flag to control recording
        self.video_path = "logs/cont_recording.mp4"
        self.concat_video_path = "logs/cont_recording_concat.mp4"
        self.video_fps = 20
        self.video_writer = None
        self.concat_video_writer = None


        self.current_step = 0
        self.phase1_yaw_thresh = 30.0
        self.current_phase = 1
        self.previous_phase = 1

        self.current_steer = 0.25
        self.reverse = True
        self.gear = -1

        self.steer_history = deque(maxlen=5)

        self.gear_change_count = 0
        self.previous_gear = -1
        self.gear_reward_handled = False

        self.world_mgr = CarlaWorldManager(
            host=self.config.get('host', 'localhost'),
            port=self.config.get('port', 2000),
            town=self.config.get('town', 'Town03'),
            synchronous=True
        )

        self.gap_length = 0.0
        self.min_gap_ratio = 1.2

        self.ego_vehicle: carla.Vehicle = self.world_mgr.spawn_ego_vehicle()

        self.sensor_manager = SensorManager(self.world_mgr.world, self.ego_vehicle, config={
            "record_bev": True,
            "bev_video_path": "logs/bev.mp4",
            "bev_fps": 20
        })

        self.world_mgr._set_spectator_to_ego()

        # Wait for valid lidar data
        ogm_1d = self.get_initial_ogm()

        self.parking_side = self.estimate_parking_side(ogm_1d, self.grid_size_width, self.grid_size_height)
        print(f"[INFO] Estimated Parking Side: {'RIGHT' if self.parking_side == 1 else 'LEFT'}")
        self.side_str = "left" if self.parking_side == 0 else "right"

        self.action_space = gym.spaces.Box(
            low=np.array([-0.5, -0.5, 0]),
            high=np.array([0.5, 1, 1]),
            shape=(3,),
            dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2 + self.grid_size_height * self.grid_size_width,),
            dtype=np.float32
        )

        # self.target_spot = self._define_parking_spot()
        self.has_collided = False
        self.successful = False

        os.makedirs("logs", exist_ok=True)
        self.log_file = open("logs/parallel_parking_log.txt", "w")

    def get_initial_ogm(self):
        # Wait for valid lidar data
        lidar_pts = None
        while lidar_pts is None:
            self.world_mgr.tick()
            lidar_pts = self.sensor_manager.get_lidar_data()

        ogm_1d = lidar_to_occupancy_grid(
            lidar_pts, grid_width=self.grid_size_width, grid_height=self.grid_size_height, resolution=0.2
        )
        return ogm_1d

    def estimate_parking_side(self, ogm, w, h):
        ogm_2d = np.reshape(ogm, (h, w))
        left_sum = np.sum(ogm_2d[:, :w // 2])
        right_sum = np.sum(ogm_2d[:, w // 2:])
        return 1 if right_sum > left_sum else 0

    def safe_destroy_and_spawn(self):
        self.sensor_manager.destroy()
        self.world_mgr.destroy_walls()
        self.world_mgr.destroy_ego_vehicle()
        self.world_mgr.destroy_boundary_vehicles()

        self.world_mgr.tick()
        time.sleep(0.25)  # Allow simulation to process

        self.ego_vehicle = self.world_mgr.spawn_ego_vehicle()
        self.sensor_manager = SensorManager(self.world_mgr.world, self.ego_vehicle)
        self._spawn_boundary_cars()
        self.gap_length = self.world_mgr.gap_length

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.episode += 1

        # Destroy and respawn the vehicles and the sensor
        self.safe_destroy_and_spawn()

        self.current_step = 0
        self.has_collided = False
        self.successful = False
        self.current_phase = 1
        self.previous_phase = 1
        self.gear_change_count = 0
        self.gear = -1
        self.previous_gear = -1

        frame = None
        lidar_pts = None
        timeout = time.time() + 1.0  # wait max 1 seconds for valid frame
        while frame is None and time.time() < timeout and lidar_pts is None:
            self.world_mgr.tick()
            frame = self.sensor_manager.get_bev_camera_frame()
            lidar_pts = self.sensor_manager.get_lidar_data()

        ogm_1d = self.get_initial_ogm()
        self.parking_side = self.estimate_parking_side(ogm_1d, self.grid_size_width, self.grid_size_height)
        print(f"[INFO] Estimated Parking Side: {'RIGHT' if self.parking_side == 1 else 'LEFT'}")
        self.side_str = "left" if self.parking_side == 0 else "right"

        frame = self.sensor_manager.get_bev_camera_frame()  # or get_camera_frame()

        if self.record_video and self.video_writer is None and frame is not None:
            h, w = frame.shape[:2]
            os.makedirs(os.path.dirname(self.video_path), exist_ok=True)
            self.video_writer = cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*'mp4v'), self.video_fps,
                                                (w, h))

        if self.record_video and self.concat_video_writer is None and frame is not None:
            ogm_vis = self._get_ogm_visual(frame)
            if ogm_vis is not None:
                concat_w = frame.shape[1] + ogm_vis.shape[1]
                self.concat_video_writer = cv2.VideoWriter(
                    self.concat_video_path, cv2.VideoWriter_fourcc(*'mp4v'),
                    self.video_fps, (concat_w, frame.shape[0])
                )

        return self._get_observation(), {}

    def step(self, action):
        self.current_step += 1
        self._apply_control(action)
        self.world_mgr.tick()

        frame = self.sensor_manager.get_bev_camera_frame()

        self.world_mgr.follow_ego_vehicle()

        obs = self._get_observation()
        gt_yaw = self._ground_truth_yaw()
        distances = self._compute_vehicle_distances()
        reward, done = self._compute_reward_and_done(distances, gt_yaw)

        self._update_phase_logic(distances, gt_yaw, parking_side=self.parking_side)

        self._log_step(action, obs, distances, gt_yaw, reward, done)


        terminated = done
        truncated = self.current_step >= self.max_steps and not terminated

        if (self.record_video) and (self.video_writer is not None) and (self.concat_video_writer is not None) and (frame is not None):

            # Concat visualization
            ogm_vis = self._get_ogm_visual(frame)
            concat_frame = cv2.hconcat([frame, ogm_vis])

            # Add episode text
            text = f"Episode {self.episode}"

            frame = put_episode_text(frame, text)
            concat_frame = put_episode_text(concat_frame, text)


            self.video_writer.write(frame)
            self.concat_video_writer.write(concat_frame)

        return obs, reward, terminated, truncated , {}

    def _apply_control(self, action):
        steer, throttle, gear_flag = action

        brake = 0.0
        if throttle <= 0:
            brake = abs(throttle)
            throttle = 0.0

        control = carla.VehicleControl()
        control.steer = max(-1, steer)
        control.throttle =  max(0, throttle)
        control.brake = max(0, brake)
        control.gear = -1 if gear_flag > 0.5 else 1
        control.reverse = (control.gear == -1)

        if self.current_phase == 2:
            control.steer = 0.0

        if self.current_phase < 3:
            control.gear = -1
            control.reverse = True



        self.previous_gear = self.gear
        self.gear = control.gear
        if self.previous_gear != self.gear:
            self.gear_change_count += 1
            self.gear_reward_handled = False

        self.steer_history.append(steer)
        self.ego_vehicle.apply_control(control)

    def _get_observation(self):
        lidar_pts = self.sensor_manager.get_lidar_data()
        ogm_1d = lidar_to_occupancy_grid(
            lidar_pts, grid_width=self.grid_size_width, grid_height=self.grid_size_height, resolution=0.2
        )

        transform = self.ego_vehicle.get_transform()
        velocity = self.ego_vehicle.get_velocity()

        yaw = transform.rotation.yaw
        speed = np.sqrt(velocity.x ** 2 + velocity.y ** 2)

        obs = np.zeros(2 + self.grid_size_height * self.grid_size_width, dtype=np.float32)
        obs[0] = yaw
        obs[1] = speed
        obs[2:] = ogm_1d

        return obs

    def _get_ogm_visual(self, frame):
        lidar_pts = self.sensor_manager.get_lidar_data()
        if lidar_pts is None:
            return None
        ogm = lidar_to_occupancy_grid(lidar_pts, self.grid_size_width, self.grid_size_height, resolution=0.2)
        ogm_2d = ogm.reshape((self.grid_size_height, self.grid_size_width)) * 255
        ogm_vis = cv2.resize(
            ogm_2d,
            (int(frame.shape[0] * self.grid_size_width / self.grid_size_height), frame.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        return cv2.cvtColor(ogm_vis.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    def _compute_reward_and_done(self, distances, gt_yaw):
        reward = 0.0
        done = False

        rear_left = distances['rear_left_dist']
        rear_right = distances['rear_right_dist']
        rear_center = distances['rear_center_dist']
        front_center = distances['front_center_dist']
        yaw_error = abs(gt_yaw)

        if self.current_phase > self.previous_phase:
            reward += 10.0 * self.current_phase
        self.previous_phase = self.current_phase

        if self.current_phase == 1:

            reward -= 0.5 * abs(yaw_error - 30.0)
            reward = (reward - 0.4 * rear_left) if self.parking_side == 1 else ( reward - 0.4 * rear_right)

        elif self.current_phase == 2:

            reward -= 0.4 * rear_center

        elif self.current_phase == 3:

            #reward -= 0.15 * rear_center
            reward -= 0.15 * yaw_error
            center_diff = abs(front_center - rear_center)
            reward -= 0.3 * center_diff

            parked, park_reward = self._check_if_parked(rear_center, front_center, yaw_error)
            if parked:
                self.successful = True
                reward += park_reward
                done = True

        sim_time = self.current_step * 0.05
        step_penalty = 0.05 * (8 if sim_time > 8 else 4 if sim_time > 4 else 2)
        reward -= step_penalty

        if len(self.steer_history) == 5:
            sign_changes = 0
            prev_sign = np.sign(self.steer_history[0])
            for val in list(self.steer_history)[1:]:
                current_sign = np.sign(val)
                if current_sign != 0 and current_sign != prev_sign:
                    sign_changes += 1
                prev_sign = current_sign

            if sign_changes >= 2:
                reward -= 2.5 * sign_changes



        if not self.gear_reward_handled and self.gear_change_count >=2:
            reward -= 5.0 * (self.gear_change_count - 1)
            self.gear_reward_handled = True


        if self._check_collision():
            reward -= 100.0
            done = True

        return reward, done

    def _check_if_parked(self, rear_center, front_center, yaw_error):
        center_diff = abs(front_center - rear_center)
        # We need to also check the hoprizontal laign in the final park
        parked = center_diff < 0.2 and abs(yaw_error) < 5.0
        reward = 0.0

        if parked:
            # Bonus inversely proportional to error (but bounded)
            yaw_bonus = max(0.0, (5.0 - abs(yaw_error)) * 2.0)
            align_bonus = max(0.0, (0.2 - center_diff) * 50.0)
            reward = 100.0 + yaw_bonus + align_bonus

        return parked, reward

    def _update_phase_logic(self, distances, gt_yaw, parking_side):
        phase2_thresh = 0.25 * 6.0

        rear_corner = distances['rear_left_dist'] if parking_side == 1 else distances['rear_right_dist']

        if self.current_phase == 1:
            if abs(gt_yaw) > self.phase1_yaw_thresh:
                self.current_phase = 2
                self.current_steer = 0.0
        elif self.current_phase == 2:
            if rear_corner < phase2_thresh:
                self.current_phase = 3
                self.current_steer = 0.35

    def _ground_truth_yaw(self):
        ego_yaw = self.ego_vehicle.get_transform().rotation.yaw
        front_yaw = self.world_mgr.npc_vehicles[0].get_transform().rotation.yaw
        return (abs(ego_yaw - front_yaw) + 180) % 360 - 180

    def _compute_vehicle_distances(self):
        def get_corners(vehicle):
            tf = vehicle.get_transform()
            extent = vehicle.bounding_box.extent
            forward = tf.get_forward_vector()
            right = tf.get_right_vector()
            loc = tf.location

            rear_c = loc - forward * extent.x
            rear_l = rear_c - right * extent.y
            rear_r = rear_c + right * extent.y

            front_c = loc + forward * extent.x
            front_l = front_c - right * extent.y
            front_r = front_c + right * extent.y

            return {
                'rear_center': rear_c, 'rear_left': rear_l, 'rear_right': rear_r,
                'front_center': front_c, 'front_left': front_l, 'front_right': front_r
            }

        def dist(a, b):
            return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

        ego_p = get_corners(self.ego_vehicle)
        rear_p = get_corners(self.world_mgr.npc_vehicles[0])
        front_p = get_corners(self.world_mgr.npc_vehicles[1])

        return {
            'rear_center_dist': dist(ego_p['rear_center'], rear_p['front_center']),
            'rear_left_dist': dist(ego_p['rear_left'], rear_p['front_left']),
            'rear_right_dist': dist(ego_p['rear_right'], rear_p['front_right']),
            'front_center_dist': dist(ego_p['front_center'], front_p['rear_center']),
            'front_left_dist': dist(ego_p['front_left'], front_p['rear_left']),
            'front_right_dist': dist(ego_p['front_right'], front_p['rear_right'])
        }

    def _check_collision(self):
        return len(self.sensor_manager.get_collision_history()) > 0


    def _spawn_boundary_cars(self):
        ego_tf = self.ego_vehicle.get_transform()
        self.world_mgr._spawn_boundary_vehicles(self.ego_vehicle.bounding_box,self.world_mgr.spawn_point, side_str=self.side_str)

        print(f"Len of spawned boundary vehicles: {len(self.world_mgr.npc_vehicles)}")

        if len(self.world_mgr.npc_vehicles) != 2:
            raise RuntimeError("[CRITICAL ❌] Failed to spawn both boundary cars after retries. Reset aborted.")


    def _define_parking_spot(self):
        center = self.world_mgr.get_parking_spot_center()
        return {'x': center.x, 'y': center.y, 'yaw': 0.0}

    def _log_step(self, action, obs, distances, yaw, reward, done):
        self.log_file.write(f"Step {self.current_step}\n")
        self.log_file.write(f"Action: {action}\n")
        self.log_file.write(f"Yaw Error: {yaw:.2f}\n")
        self.log_file.write(f"Distances: {distances}\n")
        self.log_file.write(f"Phase: {self.current_phase}\n")
        self.log_file.write(f"Gear: {self.gear}\n")
        self.log_file.write(f"Reward: {reward:.2f} | Done: {done}\n\n")

    def render(self, mode='human'):
        lidar_pts = self.sensor_manager.get_lidar_data()
        ogm_1d = lidar_to_occupancy_grid(lidar_pts, grid_width=self.grid_size_width, grid_height=self.grid_size_height, resolution=0.2)
        show_occupancy_grid(ogm_1d, grid_size=(self.grid_size_height, self.grid_size_width), window_name=f"OGM Step {self.current_step}")

    def close(self):
        if self.sensor_manager:
            self.sensor_manager.destroy()
        self.world_mgr.cleanup()
        self.log_file.close()

        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        if self.concat_video_writer:
            self.concat_video_writer.release()
            self.concat_video_writer = None

        cv2.destroyAllWindows()

