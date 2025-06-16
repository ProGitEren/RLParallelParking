import gymnasium as gym
import carla

from envs.carla_parallel_env import CarlaParallelParkingEnv

class CarlaParallelParkingDiscreteEnv(CarlaParallelParkingEnv):
    def __init__(self, config=None, grid_size=(60, 60)):
        super().__init__(config=config, grid_size=grid_size)

        self.steer_values = [-0.5, -0.4, -0.35, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5]
        self.throttle_values = [-1.0, -0.5, -0.2, 0.0, 0.2, 0.3, 0.5]
        self.gear_values = [0, 1]  # 0 = forward, 1 = reverse

        self.discrete_actions = []
        for steer in self.steer_values:
            for throttle in self.throttle_values:
                for gear in self.gear_values:
                    self.discrete_actions.append((steer, throttle, gear))

        self.action_space = gym.spaces.Discrete(len(self.discrete_actions))
        self.log_file = open("logs/discrete_parking_log.txt", "w")

    def step(self, action_idx):
        self.current_step += 1
        steer, throttle, gear_flag = self.discrete_actions[action_idx]
        self._apply_control(steer, throttle, gear_flag)
        self.world_mgr.tick()

        obs = self._get_observation()
        gt_yaw = self._ground_truth_yaw()
        distances = self._compute_vehicle_distances()
        reward, done = self._compute_reward_and_done(distances, gt_yaw)
        self._update_phase_logic(distances, gt_yaw, self.parking_side)
        self._log_step((steer, throttle, gear_flag), obs, distances, gt_yaw, reward, done)

        if self.current_step >= self.max_steps:
            done = True

        return obs, reward, done, {}

    def _apply_control(self, steer, throttle, gear_flag):
        brake = 0.0
        if throttle <= 0:
            brake = abs(throttle)
            throttle = 0.0

        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)
        control.gear = -1 if gear_flag == 1 else 1
        control.reverse = (control.gear == -1)

        if self.current_phase == 2:
            control.steer = 0.0

        if control.gear != self.gear:
            self.gear_change_count += 1
        self.previous_gear = self.gear
        self.gear = control.gear
        self.steer_history.append(steer)

        self.ego_vehicle.apply_control(control)
