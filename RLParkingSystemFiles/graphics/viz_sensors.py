# graphics/viz_sensors.py

import numpy as np
import matplotlib.pyplot as plt
import cv2

def show_lidar_points(lidar_pts, ax=None):
    """
    Plot a 2D scatter of LiDAR points projected onto X-Y plane.
    This is a basic approach ignoring color/intensity or 3D overhead.
    """
    if lidar_pts is None or len(lidar_pts) == 0:
        print("No LiDAR data to display.")
        return

    x = lidar_pts[:, 0]
    y = lidar_pts[:, 1]

    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(x, y, s=1, c='blue')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect('equal', 'box')
    ax.set_title("LiDAR XY Projection")
    plt.show(block=False)

def show_camera_frame(cam_frame, window_name="Camera"):
    """
    Display an RGB camera frame using OpenCV.
    """
    if cam_frame is None:
        print("No camera frame to display.")
        return

    # cam_frame is likely a (H,W,3) BGR or RGB array.
    # If it was BGRA, we stripped alpha.
    # If it's in BGR, we can show directly. If it's RGB, we might convert:
    # cam_frame = cv2.cvtColor(cam_frame, cv2.COLOR_RGB2BGR)

    cv2.imshow(window_name, cam_frame)
    cv2.waitKey(1)  # 1 ms delay to allow window refresh