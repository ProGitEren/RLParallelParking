import carla
import weakref
import numpy as np
import cv2
from networkx.lazy_imports import attach
import os


class SensorManager:
    """
        Attaches LIDAR + camera sensors to the ego vehicle.
        Provides get_lidar_data(), get_camera_frame() for the RL env to read.
    """

    def __init__(self, world, vehicle, config=None):
        self.world = world
        self.vehicle = vehicle
        self.config = config or {}

        self.lidar_data = None
        self.camera_frame = None

        self._lidar_actor = None
        self._camera_actor = None

        self.bev_camera_frame = None
        self._bev_camera_actor = None

        self.enable_bev_recording = self.config.get("record_bev", False)
        self.bev_video_path = self.config.get("bev_video_path", "bev_output.mp4")
        self.bev_fps = self.config.get("bev_fps", 20)
        self.bev_video_writer = None

        self._setup_lidar()
        self._setup_camera()
        self._setup_bev_camera()

        self.collision_history = []
        self._collision_sensor = None
        self._setup_collision_sensor()



    def _setup_collision_sensor(self):
        collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        transform = carla.Transform(carla.Location(x=0.0, z=0.0))  # or any valid location
        self._collision_sensor = self.world.spawn_actor(collision_bp, transform, attach_to=self.vehicle)

        weak_self = weakref.ref(self)
        self._collision_sensor.listen(lambda event: SensorManager._collision_callback(weak_self, event))

    def _setup_lidar(self):
        lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '20')
        lidar_bp.set_attribute('rotation_frequency', '20')
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('points_per_second', '150000')

        spawn_transform = carla.Transform(carla.Location(x=0.0, z=2.5))
        self._lidar_actor = self.world.spawn_actor(lidar_bp, spawn_transform, attach_to=self.vehicle)

        weak_self = weakref.ref(self)
        self._lidar_actor.listen(lambda data: SensorManager._lidar_callback(weak_self, data))

    def _setup_camera(self):
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1280')
        camera_bp.set_attribute('image_size_y', '720')
        camera_bp.set_attribute('fov', '90')

        # Example: rear-facing camera
        spawn_transform = carla.Transform(carla.Location(x=-2.0, z=2.3), carla.Rotation(pitch=0, yaw=180, roll=0))
        self._camera_actor = self.world.spawn_actor(camera_bp, spawn_transform, attach_to=self.vehicle)

        weak_self = weakref.ref(self)
        self._camera_actor.listen(lambda data: SensorManager._camera_callback(weak_self, data))

    def _setup_bev_camera(self):
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1280')
        camera_bp.set_attribute('image_size_y', '720')
        camera_bp.set_attribute('fov', '90')  # Narrower FOV to avoid distortion

        # Top-down view from above the vehicle
        spawn_transform = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=25.0),
            carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
        )

        self._bev_camera_actor = self.world.spawn_actor(
            camera_bp, spawn_transform, attach_to=self.vehicle)

        """self.bev_video_writer = None
        if self.config.get("record_bev"):
            self.bev_video_writer = cv2.VideoWriter(
                self.config.get("bev_video_path", "logs/bev.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.config.get("bev_fps", 20),
                (800, 800)
            )"""

        weak_self = weakref.ref(self)
        self._bev_camera_actor.listen(
            lambda data: SensorManager._bev_camera_callback(weak_self, data)
        )

    @staticmethod
    def _collision_callback(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.collision_history.append(event)

    @staticmethod
    def _lidar_callback(weak_self, data):
        self = weak_self()
        if not self:
            return

        pts = np.frombuffer(data.raw_data, dtype=np.float32)
        pts = np.reshape(pts, (len(pts)//4, 4))
        self.lidar_data = pts

    @staticmethod
    def _camera_callback(weak_self, data):
        self = weak_self()
        if not self:
            return
        array = np.frombuffer(data.raw_data, dtype=np.uint8)
        array = np.reshape(array, (data.height, data.width, 4))
        # Convert BGRA to BGR or RGB
        self.camera_frame = array[:,:,:3] # ignoring alfa

    @staticmethod
    def _bev_camera_callback(weak_self, data):
        self = weak_self()
        if not self:
            return

        array = np.frombuffer(data.raw_data, dtype=np.uint8)
        array = np.reshape(array, (data.height, data.width, 4))
        frame = array[:, :, :3]  # Drop alpha

        self.bev_camera_frame = frame

        if self.enable_bev_recording:
            if self.bev_video_writer is None:
                # Ensure directory exists
                os.makedirs(os.path.dirname(self.bev_video_path), exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.bev_video_writer = cv2.VideoWriter(
                    self.bev_video_path,
                    fourcc,
                    self.bev_fps,
                    (data.width, data.height)
                )
            print("[VIDEO] Writing frame to video...")
            self.bev_video_writer.write(frame)

    def get_lidar_data(self):
        return self.lidar_data

    def get_camera_frame(self):
        return self.camera_frame

    def get_bev_camera_frame(self):
        return self.bev_camera_frame

    def get_collision_history(self):
        return self.collision_history

    def destroy(self):

        if self._lidar_actor:
            self._lidar_actor.stop()
            self._lidar_actor.destroy()

        if self._camera_actor:
            self._camera_actor.stop()
            self._camera_actor.destroy()

        if self._bev_camera_actor:
            self._bev_camera_actor.stop()
            self._bev_camera_actor.destroy()

        if self.bev_video_writer:
            self.bev_video_writer.release()
            print("[VIDEO] Releasing video writer.")



