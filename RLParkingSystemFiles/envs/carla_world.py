from distutils.spawn import spawn

import carla
import random
import time

class CarlaWorldManager:
    """
        Manages CARLA client, world, and the main ego vehicle
        spawn/destroy logic.
        Provides synchronous stepping for stable TL interaction.
    """

    def __init__(self, host='localhost', port=2000, town = 'Town03', synchronous=True, delta_seconds=0.05):
        self.host = host
        self.port = port
        self.town = town
        self.synchronous = synchronous
        self.delta_seconds = delta_seconds

        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(10.0)

        if not hasattr(self, 'world') or self.world.get_map().name != self.town:
            self.world = self.client.load_world(self.town)

        self.original_settings = self.world.get_settings()

        settings = self.world.get_settings()
        settings.fixed_delta_seconds = self.delta_seconds
        settings.synchronous_mode = self.synchronous
        self.world.apply_settings(settings)

        self.blueprint_lib: carla.BlueprintLibrary = self.world.get_blueprint_library()
        self.ego_vehicle:carla.Vehicle = None
        self.spawn_point = None

        self.parking_spot_center = None
        self.npc_vehicles = []
        self.parking_side = None  # 0 = left, 1 = right
        self.side_str = "right"
        self.gap_length = 0.0
        self.lateral_offset = 0.0

        self.wall_actors = []

    def _find_roadside_spawn_point(self):
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)  # <- Ensure randomness

        for sp in spawn_points:
            wp = self.world.get_map().get_waypoint(sp.location)
            if wp and wp.lane_type == carla.LaneType.Driving:
                right_wp = wp.get_right_lane()
                if right_wp and right_wp.lane_type == carla.LaneType.Sidewalk:
                    self.spawn_point = sp
                    return sp

        print("[WARNING] No roadside spawn point found with sidewalk. Using default driving lane.")
        for sp in spawn_points:
            wp = self.world.get_map().get_waypoint(sp.location)
            if wp and wp.lane_type == carla.LaneType.Driving:
                self.spawn_point = sp
                return sp

        print("[FATAL] No valid spawn point found.")
        self.spawn_point = random.choice(spawn_points)
        return self.spawn_point

    def spawn_ego_vehicle(self, transform=None):
        """Spawn or re-spawn the ego vehicle"""
        if self.ego_vehicle:
            self.destroy_ego_vehicle()

        vehicle_bp = self.blueprint_lib.find('vehicle.ford.mustang')
        spawn_point = self._find_roadside_spawn_point()
        self.ego_vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

        print(f"Ego Vehicle: {self.ego_vehicle.get_transform().location}")
        return self.ego_vehicle

    def _set_spectator_to_ego(self):
        spectator = self.world.get_spectator()
        transform = self.ego_vehicle.get_transform()
        spectator_location = transform.location + carla.Location(x=-6, z=3)
        spectator_rotation = carla.Rotation(pitch=-15, yaw=transform.rotation.yaw)
        spectator.set_transform(carla.Transform(spectator_location, spectator_rotation))

    def follow_ego_vehicle(self):

        spectator = self.world.get_spectator()
        transform = self.ego_vehicle.get_transform()
        cam_location = transform.location + carla.Location(z=15)
        cam_rotation = carla.Rotation(pitch=-90, yaw=transform.rotation.yaw)
        spectator.set_transform(carla.Transform(cam_location, cam_rotation))

    def get_dynamic_gap_and_offset(self, ego_vehicle: carla.Vehicle, side="right"):
        extent = ego_vehicle.bounding_box.extent
        base_gap = extent.x * 2
        base_width = extent.y * 2

        gap_length = base_gap * random.uniform(1.2, 1.4)
        self.gap_length = gap_length

        lateral_base = base_width * random.uniform(1.05, 1.25)
        lateral_offset = lateral_base if side == "right" else -lateral_base
        return gap_length, lateral_offset

    def _spawn_boundary_vehicles(self, bbox:carla.BoundingBox, ego_tf: carla.Transform, side_str):
        self.destroy_boundary_vehicles()
        vehicle_bp = self.blueprint_lib.find("vehicle.ford.mustang")
        half_length = bbox.extent.x


        for attempt in range(5):

            print(f"[INFO] Attempt {attempt + 1} to spawn boundary cars...")
            for npc in self.npc_vehicles:
                npc.destroy()
            self.npc_vehicles = []

            self.parking_side = random.choice([0, 1])
            side_str = "right" if self.parking_side else "left"
            gap_length, lateral_offset = self.get_dynamic_gap_and_offset(self.ego_vehicle, side=side_str)
            self.gap_length = gap_length
            self.lateral_offset = lateral_offset

            forward_shift = random.uniform(0.0, 0.05)
            front_offset = carla.Location(x=-forward_shift, y=lateral_offset, z=0.3)
            rear_offset = carla.Location(x=-forward_shift - gap_length - 2 * half_length, y=lateral_offset, z=0.3)

            rear_tf = carla.Transform(ego_tf.transform(rear_offset), ego_tf.rotation)
            front_tf = carla.Transform(ego_tf.transform(front_offset), ego_tf.rotation)

            print(f"Rear TF: {rear_tf.location}, Front TF: {front_tf.location}")

            rear_car = self.world.try_spawn_actor(vehicle_bp, rear_tf)
            front_car = self.world.try_spawn_actor(vehicle_bp, front_tf)

            if rear_car and front_car:
                self.npc_vehicles = [rear_car, front_car]
                print("[SPAWNED ✅] Rear and Front boundary cars.")
                rear_car.set_simulate_physics(False)
                front_car.set_simulate_physics(False)

                self._spawn_wall_near_gap(ego_tf, side_str)

                return

            print(f"[RETRY {attempt + 1}] Failed to spawn both boundary cars.")


        print("[ERROR ❌] Could not spawn both boundary vehicles after retries.")
        self.npc_vehicles = []

    def _spawn_wall_near_gap(self, ego_tf, side_str):
        if len(self.npc_vehicles) != 2:
            print("[WARNING] Wall not spawned: boundary vehicles not available.")
            return

        wall_bp = self.blueprint_lib.find("static.prop.streetbarrier")

        # Get rear vehicle bounding box width (use it to determine wall lateral offset)
        rear_car = self.npc_vehicles[0]
        npc_extent = rear_car.bounding_box.extent
        margin = 0.7
        lateral_shift = abs(self.lateral_offset) + npc_extent.y + margin
        lateral_offset = lateral_shift if side_str == "right" else -lateral_shift

        # Compute midpoint along the x-axis in the spawn logic
        half_length = rear_car.bounding_box.extent.x
        mid_forward_offset = - (self.gap_length / 2 + half_length)

        # Wall offset relative to ego
        wall_center_offset = carla.Location(
            x=mid_forward_offset,
            y=lateral_offset,
            z=0.3
        )
        wall_forward_offset = carla.Location(
            x=mid_forward_offset + 1.5,
            y=lateral_offset,
            z=0.3
        )
        wall_rear_offset = carla.Location(
            x=mid_forward_offset - 1.5,
            y=lateral_offset,
            z=0.3
        )

        wall_center_tf = carla.Transform(ego_tf.transform(wall_center_offset), ego_tf.rotation)
        wall_forward_tf = carla.Transform(ego_tf.transform(wall_forward_offset), ego_tf.rotation)
        wall_rear_tf = carla.Transform(ego_tf.transform(wall_rear_offset), ego_tf.rotation)

        wall_center = self.world.try_spawn_actor(wall_bp, wall_center_tf)
        wall_forward = self.world.try_spawn_actor(wall_bp, wall_forward_tf)
        wall_rear = self.world.try_spawn_actor(wall_bp, wall_rear_tf)
        if wall_center and wall_forward and wall_rear:
            wall_center.set_simulate_physics(False)
            wall_forward.set_simulate_physics(False)
            wall_rear.set_simulate_physics(False)
            self.wall_actors = [wall_center, wall_forward, wall_rear]
            print(f"[SPAWNED ✅] Center Wall at {wall_center_tf.location} on side: {side_str}")
            print(f"[SPAWNED ✅] Forward Wall at {wall_forward_tf.location} on side: {side_str}")
            print(f"[SPAWNED ✅] Rear Wall at {wall_rear_tf.location} on side: {side_str}")
        else:
            print("[WARNING] Walls could not be spawned.")


    def destroy_ego_vehicle(self):
        """Destroy the ego vehicle if it exists"""
        if self.ego_vehicle:
            self.ego_vehicle.destroy()
            self.ego_vehicle = None

    def get_parking_spot_center(self):
        return self.parking_spot_center

    def tick(self):
        if self.synchronous:
            self.world.tick()
        else:
            time.sleep(self.delta_seconds)

    def cleanup(self):
        self.destroy_ego_vehicle()
        self.destroy_boundary_vehicles()
        self.destroy_walls()
        self.world.apply_settings(self.original_settings)


    def destroy_walls(self):
        if self.wall_actors:
            for wall in self.wall_actors:
                wall.destroy()

            self.wall_actors = []

    def destroy_boundary_vehicles(self):
        for vehicle in self.npc_vehicles:
            if vehicle:
                vehicle.destroy()
        self.npc_vehicles = []


if __name__ == "__main__":
    model = CarlaWorldManager()
    print(model)
    print("Model initialized successfully!")
