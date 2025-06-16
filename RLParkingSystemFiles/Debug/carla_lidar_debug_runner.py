import carla
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from collections import deque

from sensors.sensor_manager import SensorManager
from sensors.data_fusion import lidar_to_occupancy_grid
from graphics.viz_occupancy import show_occupancy_grid


class CarlaDebugRunner:
    def __init__(self, host='localhost', port=2000, town='Town03', grid_width=60, grid_height=60):
        self.client = carla.Client(host, port)
        self.client.set_timeout(20.0)
        self.world: carla.World = self.client.load_world(town)
        self.original_settings = self.world.get_settings()

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        self.bp_lib = self.world.get_blueprint_library()
        self.ego_vehicle = None
        self.sensor_manager = None
        self.npc_vehicles = []
        self.current_step = 0

        self.grid_height = grid_height
        self.grid_width = grid_width

        self.parking_side = None
        self.current_phase = 1
        self.previous_phase = 1
        self.current_steer = 0.25
        self.reverse = True
        self.gear = -1
        self.phase1_yaw_thresh = 30.0
        self.gap_length = 0.0

        self.steer_history = deque(maxlen=5)
        self.gear_change_count = 0
        self.previous_gear = -1

    def setup(self):
        print("[INFO] Spawning ego vehicle...")
        vehicle_bp = self.bp_lib.find('vehicle.ford.mustang')
        spawn_point = self._find_roadside_spawn_point()
        self.ego_vehicle: carla.Vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

        print(f"Ego vehicle: {self.ego_vehicle.get_transform().location}")

        print("[INFO] Attaching sensors...")
        self.sensor_manager = SensorManager(self.world, self.ego_vehicle)

        print("[INFO] Spawning boundary cars with parking gap...")
        ego_tf = self.ego_vehicle.get_transform()

        success = False
        for attempt in range(5):
            print(f"[INFO] Attempt {attempt + 1} to spawn boundary cars...")
            for npc in self.npc_vehicles:
                npc.destroy()
            self.npc_vehicles = []

            side_choice = random.choice(["left", "right"])
            gap_length, lateral_offset = self.get_dynamic_gap_and_offset(self.ego_vehicle, side=side_choice)
            self.gap_length = gap_length

            self._spawn_boundary_cars_with_gap(gap_length, self.ego_vehicle.bounding_box, spawn_point, lateral_offset)

            if len(self.npc_vehicles) == 2:
                success = True
                self.parking_side = 1 if side_choice == "right" else 0
                print("[INFO] Successfully spawned both boundary cars!")
                break

        if not success:
            print("[ERROR ❌] Failed to spawn boundary cars after multiple attempts.")

        self._set_spectator_to_ego()

    def _find_roadside_spawn_point(self):
        spawn_points = self.world.get_map().get_spawn_points()
        for sp in spawn_points:
            wp = self.world.get_map().get_waypoint(sp.location)
            right_wp = wp.get_right_lane()
            if right_wp and right_wp.lane_type == carla.LaneType.Sidewalk:
                return sp
        print("[WARNING] No roadside spawn point found, using default.")
        return random.choice(spawn_points)

    def _check_collision(self):
        return len(self.sensor_manager.get_collision_history()) > 0

    def _spawn_boundary_cars_with_gap(self, gap_length, bounding_box: carla.BoundingBox, ego_tf: carla.Transform, lateral_offset=2.2):
        vehicle_bp = self.bp_lib.find("vehicle.ford.mustang")
        half_length = bounding_box.extent.x
        forward_shift = random.uniform(0.0, 0.05)

        rear_offset = carla.Location(x=-forward_shift, y=lateral_offset, z=0.3)
        front_offset = carla.Location(x=-forward_shift - gap_length - 2 * half_length, y=lateral_offset, z=0.3)

        rear_tf = carla.Transform(ego_tf.transform(rear_offset), ego_tf.rotation)
        front_tf = carla.Transform(ego_tf.transform(front_offset), ego_tf.rotation)

        print(f"Rear TF: {rear_tf.location}, Front TF: {front_tf.location}")

        rear_car = self.world.try_spawn_actor(vehicle_bp, rear_tf)
        if rear_car:
            self.npc_vehicles.append(rear_car)
            print("[SPAWNED ✅] Rear boundary car.")
        else:
            print("[FAILED ❌] Rear boundary car spawn failed.")

        front_car = self.world.try_spawn_actor(vehicle_bp, front_tf)
        if front_car:
            self.npc_vehicles.append(front_car)
            print("[SPAWNED ✅] Front boundary car.")
        else:
            print("[FAILED ❌] Front boundary car spawn failed.")

    def run(self, num_steps=100):
        plt.figure(figsize=(6, 3))
        lidar_pts = None
        while lidar_pts is None:
            self.world.tick()
            lidar_pts = self.sensor_manager.get_lidar_data()

        ogm_1d = lidar_to_occupancy_grid(lidar_pts, self.grid_width, self.grid_height, resolution=0.2)
        self.parking_side = self.estimate_parking_side(ogm_1d, self.grid_width, self.grid_height)
        self.current_steer = 0.35 if self.parking_side == 1 else -0.35

        for step in range(num_steps):
            self.current_step = step
            print(f"[STEP {step}]")

            if len(self.npc_vehicles) >= 2:
                gt_yaw = self.ground_truth_yaw(self.ego_vehicle, self.npc_vehicles[0])
                distances = self.compute_vehicle_distances(self.ego_vehicle, self.npc_vehicles[0], self.npc_vehicles[1])

                print(f"Yaw Error: {abs(gt_yaw):.2f}°")
                print(f"Rear L/C/R: {distances['rear_left_dist']:.2f}, {distances['rear_center_dist']:.2f}, {distances['rear_right_dist']:.2f}")

                if any(distances[key] < 0.45 for key in ['rear_left_dist', 'rear_center_dist', 'rear_right_dist']):
                    self.gear_change_count += 1
                    self.gear = 1
                    self.reverse = False
                    self.current_steer = -0.35 if self.parking_side == 0 else 0.35
                    print(f"[GEAR CHANGE] ➔ FORWARD with steer {self.current_steer:.2f}")

                self._apply_reverse_control()

                self.steer_history.append(self.current_steer)
                self.update_phase_logic(distances, gt_yaw, self.gap_length)
                reward, done = self.compute_reward_and_done(distances, gt_yaw)
                print(f"[REWARD] {reward:.2f} | Done: {done}")

                if done:
                    print("[DONE FLAG] Ending episode.")
                    break

            else:
                self._apply_reverse_control()

            self.world.tick()
            self._follow_ego(step)

            lidar_pts = self.sensor_manager.get_lidar_data()
            if lidar_pts is not None:
                ogm_1d = lidar_to_occupancy_grid(lidar_pts, 60, 60, resolution=0.2)
                show_occupancy_grid(ogm_1d, grid_size=(60, 60), window_name=f"OGM Step {step}")

        self.cleanup()

    def _apply_reverse_control(self):
        control = carla.VehicleControl()
        control.throttle = 0.2
        control.reverse = self.reverse
        control.gear = self.gear
        control.steer = self.current_steer
        self.ego_vehicle.apply_control(control)

    def _set_spectator_to_ego(self):
        spectator = self.world.get_spectator()
        ego_tf = self.ego_vehicle.get_transform()
        cam_location = ego_tf.location + carla.Location(z=15)
        cam_rotation = carla.Rotation(pitch=-90, yaw=ego_tf.rotation.yaw)
        spectator.set_transform(carla.Transform(cam_location, cam_rotation))

    def _follow_ego(self, step):

        self._set_spectator_to_ego()

    def ground_truth_yaw(self, ego_vehicle: carla.Vehicle, front_vehicle: carla.Vehicle):
        ego_yaw = ego_vehicle.get_transform().rotation.yaw
        front_yaw = front_vehicle.get_transform().rotation.yaw
        return (abs(ego_yaw - front_yaw) + 180) % 360 - 180

    def compute_vehicle_distances(self, ego, front, rear):
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

        ego_p, rear_p, front_p = get_corners(ego), get_corners(rear), get_corners(front)
        return {
            'rear_center_dist': dist(ego_p['rear_center'], rear_p['front_center']),
            'rear_left_dist': dist(ego_p['rear_left'], rear_p['front_left']),
            'rear_right_dist': dist(ego_p['rear_right'], rear_p['front_right']),
            'front_center_dist': dist(ego_p['front_center'], front_p['rear_center']),
            'front_left_dist': dist(ego_p['front_left'], front_p['rear_left']),
            'front_right_dist': dist(ego_p['front_right'], front_p['rear_right'])
        }

    def estimate_parking_side(self, ogm, w, h):
        ogm_2d = np.reshape(ogm, (h, w))
        left_sum = np.sum(ogm_2d[:, :w // 2])
        right_sum = np.sum(ogm_2d[:, w // 2:])
        return 1 if right_sum > left_sum else 0

    def get_dynamic_gap_and_offset(self, ego_vehicle: carla.Vehicle, side="right"):
        extent = ego_vehicle.bounding_box.extent
        base_gap = extent.x * 2
        base_width = extent.y * 2

        gap_length = base_gap * random.uniform(1.2, 1.4)
        lateral_base = base_width * random.uniform(1.05, 1.25)
        lateral_offset = lateral_base if side == "right" else -lateral_base
        return gap_length, lateral_offset

    def update_phase_logic(self, distances, gt_yaw, gap_length):
        phase2_rear_dist_thresh = 0.25 * gap_length
        rear_center_left = distances['rear_left_dist']

        if self.current_phase == 1:
            if abs(gt_yaw) > self.phase1_yaw_thresh:
                print("[PHASE 1 ➔ 2] Switching to Phase 2")
                self.current_phase = 2
                self.current_steer = 0.0
        elif self.current_phase == 2:
            if rear_center_left < phase2_rear_dist_thresh:
                print("[PHASE 2 ➔ 3] Switching to Phase 3")
                self.current_phase = 3
                self.current_steer = 0.35 if self.parking_side == 0 else -0.35
                print(f"[INFO] Counter-steer applied: {self.current_steer:.2f}")

    def compute_reward_and_done(self, distances, gt_yaw):
        reward = 0.0
        done = False
        rear_left = distances['rear_left_dist']
        rear_center = distances['rear_center_dist']
        front_center = distances['front_center_dist']
        yaw_error = abs(gt_yaw)

        if hasattr(self, "previous_phase") and self.current_phase > self.previous_phase:
            reward += 10.0 * self.current_phase
        self.previous_phase = self.current_phase

        if self.current_phase == 1:
            reward -= 0.1 * abs(yaw_error - 30.0)
            reward -= 0.05 * rear_left
        elif self.current_phase == 2:
            reward -= 0.2 * rear_center
        elif self.current_phase == 3:
            reward -= 0.15 * rear_center
            reward -= 0.15 * yaw_error
            center_diff = abs(front_center - rear_center)
            reward -= 0.1 * center_diff

            if center_diff < 0.2 and yaw_error < 5.0:
                alignment_bonus = (10.0 - yaw_error) * 2.0
                reward += 100.0 + alignment_bonus
                done = True
                print(f"[SUCCESS] Alignment Bonus: {alignment_bonus:.2f}")

        sim_time = self.current_step * 0.05
        step_penalty = 0.01
        if sim_time > 8:
            step_penalty *= 4
        elif sim_time > 4:
            step_penalty *= 2
        reward -= step_penalty

        if len(self.steer_history) >= 5:
            diffs = [abs(self.steer_history[i] - self.steer_history[i - 1]) for i in range(1, len(self.steer_history))]
            avg_diff = sum(diffs) / len(diffs)
            reward -= 0.05 * avg_diff

        reward -= 5.0 * self.gear_change_count

        if self._check_collision():
            reward -= 100.0
            done = True
            print("[COLLISION] Ending episode.")

        return reward, done

    def cleanup(self):
        if self.sensor_manager:
            self.sensor_manager.destroy()
        self.world_mgr.cleanup()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    runner = CarlaDebugRunner()
    runner.setup()
    runner.run(num_steps=500)
    print("Process Finished!")
    runner.cleanup()
    print("Environment Cleaned!")

