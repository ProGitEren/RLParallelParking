import numpy as np
import cv2
from networkx.algorithms.bipartite.basic import density

# Tried for removing the noise
INTENSITY_THRESHOLD = 0.0

DENSITY_THRESHOLD = 1

def lidar_to_occupancy_grid(lidar_pts, grid_width=60, grid_height=30, resolution=0.2):
    """
    Create a 2D occupancy grid from 3D LiDAR points in the ego-vehicle's frame.

    :param lidar_pts: N x 4 array (x, y, z, intensity)
    :param grid_width: number of cells in the y-direction
    :param grid_height: number of cells in the x-direction
    :param resolution: meters per cell
    :return: A flattened 1D array of length (grid_height * grid_width),
             with 1.0 for occupied, 0.0 for free.
    """

    """
    X: longitude vehicle in CARLA --> Height
    Y: Width vehicle in CARLA --> Width 
    ogm --> np.array(Height, Width)
    """
    if lidar_pts is None or len(lidar_pts) == 0:
        return np.zeros(grid_height * grid_width, dtype=np.uint8)

    half_width_m = (grid_width * resolution) / 2.0
    half_height_m = (grid_height * resolution) / 2.0

    counts = np.zeros((grid_height, grid_width), dtype=np.float32)

    intensities = []

    for pt in lidar_pts:
        x, y, z, intensity = pt
        intensities.append(intensity)

        if abs(y) > half_width_m or abs(x) > half_height_m:
            continue

        origin_y = grid_width // 2
        origin_x = grid_height // 2
        cell_y = int((y / resolution) + origin_y)
        cell_x = int((x / resolution) + origin_x)

        if 0 <= cell_x < grid_height and 0 <= cell_y < grid_width and intensity > INTENSITY_THRESHOLD:
            counts[cell_x, cell_y] += 1  # Mark occupied

    # print(f"Min density count: {np.min(counts)}, Max density count: {np.max(counts)}")

    # ogm = (counts >= DENSITY_THRESHOLD).astype(np.float32)
    #blurred = cv2.GaussianBlur(grid, (3, 3), 0)
    #return (blurred > 0.3).astype(np.float32).flatten()
    counts_2d = counts.reshape((grid_height, grid_width))
    # blurred = cv2.GaussianBlur(ogm_2d, (3, 3), 0)
    # binary = (blurred > 0.2).astype(np.uint8)
    ogm_2d = (counts_2d >= 3).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    blocky = cv2.morphologyEx(ogm_2d, cv2.MORPH_CLOSE, kernel)
    return blocky.flatten()




def fuse_lidar_camera(lidar_pts, cam_frame):

    grid_ld = lidar_to_occupancy_grid(lidar_pts, grid_size=20, resolution=0.2)
    return grid_ld
