def estimate_yaw_from_camera(self, prev_img, curr_img, debug=False):
    if prev_img is None or curr_img is None:
        return None

    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB.create(2000)
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)

    if des1 is None or des2 is None:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) < 10:
        return None

    # Filter for ground points only (e.g., bottom half of image)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    height = prev_gray.shape[0]
    ground_mask = pts1[:, 1] > height * 0.6
    pts1 = pts1[ground_mask]
    pts2 = pts2[ground_mask]

    if len(pts1) < 8:
        return None

    # Estimate homography
    H, status = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
    if H is None:
        return None

    # Extract rotation from homography (assuming flat ground)
    # H ≈ K * [R | t] * K^-1, so we can factor out rotation
    # Using SVD approximation for 2x2 rotation
    R_approx = H[0:2, 0:2]
    U, _, VT = np.linalg.svd(R_approx)
    R = np.dot(U, VT)

    # Estimate yaw angle from rotation matrix
    yaw_rad = math.atan2(R[1, 0], R[0, 0])
    yaw_deg = np.degrees(yaw_rad)

    if debug:
        print("[HOMOGRAPHY] H:\n", H)
        print("[HOMOGRAPHY] R (approx):\n", R)
        print("[HOMOGRAPHY] Estimated yaw (deg):", yaw_deg)

    return yaw_deg


def estimate_yaw_diff(self, ego_vehicle, boundary_vehicle, ogm_1d, yaw_history, grid_width, grid_height):
    """Main interface for yaw difference estimation"""
    # Get ground truth for validation
    ego_yaw = ego_vehicle.get_transform().rotation.yaw
    boundary_yaw = boundary_vehicle.get_transform().rotation.yaw
    actual_diff = (ego_yaw - boundary_yaw + 180) % 360 - 180

    # Run RANSAC estimation
    est_diff, smoothed_est = self.ransac_yaw_estimation(
        ogm_1d,
        ego_vehicle.bounding_box.extent.x,
        ego_vehicle.bounding_box.extent.y,
        yaw_history=yaw_history,
        grid_height=grid_height,
        grid_width=grid_width
    )

    return {
        'estimated_raw': est_diff,
        'estimated_smoothed': smoothed_est,
        'actual_diff': actual_diff
    }


def improved_hough_yaw_estimation(self, ogm_1d, ego_extent_x, ego_extent_y, ego_yaw, boundary_yaw,
                                  resolution=0.1, grid_height=120, grid_width=60,
                                  prev_yaw_history=None, prev_est_yaw=None):
    ogm_2d = ogm_1d.reshape((grid_height, grid_width))

    center_row = grid_height // 2
    center_col = grid_width // 2
    ex_cells = int(ego_extent_x / resolution)
    ey_cells = int(ego_extent_y / resolution)

    roi_start_row = center_row + ex_cells - 2
    roi_end_row = grid_height
    roi_start_col = center_col + ey_cells - 2
    roi_end_col = grid_width

    roi = ogm_2d[roi_start_row:roi_end_row, roi_start_col:roi_end_col]
    roi = (roi * 255).astype(np.uint8)

    edges = cv2.Canny(roi, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=15)

    actual_yaw_diff = (ego_yaw - boundary_yaw + 180) % 360 - 180

    if lines is None:
        # Return previous estimate if available
        print("[WARN] No Hough lines detected")
        return prev_est_yaw if prev_est_yaw is not None else 0.0, actual_yaw_diff

    # Filter good angle lines
    thetas = []
    for line in lines:
        rho, theta = line[0]
        theta_deg = np.degrees(theta)
        if 75 < theta_deg < 105:  # Narrower range for alignment
            thetas.append(theta_deg)

    if not thetas:
        return prev_est_yaw if prev_est_yaw is not None else 0.0, actual_yaw_diff

    # Histogram binning
    hist, bin_edges = np.histogram(thetas, bins=20)
    best_bin = np.argmax(hist)
    filtered = [t for t in thetas if bin_edges[best_bin] <= t <= bin_edges[best_bin + 1]]

    mean_theta = np.mean(filtered)
    est_yaw_diff = 90.0 - mean_theta  # Assuming vehicle direction is horizontal

    # Smooth with history
    if prev_yaw_history is not None:
        prev_yaw_history.append(est_yaw_diff)
        if len(prev_yaw_history) > 5:
            prev_yaw_history.pop(0)
        est_yaw_diff = np.mean(prev_yaw_history)

    return est_yaw_diff, actual_yaw_diff


def estimate_rear_clearance(self, ego_vehicle: carla.Vehicle, grid_1d, resolution=0.1, grid_height=80, grid_width=40):
    """
        Estimates vertical clearance behind the ego vehicle using the occupancy grid.
        Parameters:
            ego_vehicle: carla.Vehicle object (to access bounding box)
            grid_1d: 1D flattened occupancy grid
            resolution: size of each cell (meters)
            grid_height: number of rows (Y)
            grid_width: number of columns (X)
        Returns:
            (min_dist, avg_dist): minimum and average distance (in meters) from ego rear edge to nearest obstacle
    """

    grid = np.reshape(grid_1d, (grid_height, grid_width))

    # Ego vehicle is centered
    center_row = grid_height // 2
    center_col = grid_width // 2

    # Get half-vehicle width (in cols) and rear offset (in rows)
    extent_y = ego_vehicle.bounding_box.extent.y
    extent_x = ego_vehicle.bounding_box.extent.x
    half_width_cells = int(extent_y / resolution)
    half_height_cells = int(extent_x / resolution)

    # Define the rear end row of the vehicle
    rear_row = center_row - int(extent_x / resolution)

    # Define columns for left-to-right width of the vehicle
    left_col = center_col - half_width_cells
    right_col = center_col + half_width_cells

    distances = []
    for col in range(left_col, right_col + 1):
        for row in range(rear_row, 0, -1):
            if grid[row, col] > 0.5:
                dy = rear_row - row
                distances.append(dy * resolution)
                break
    if not distances:
        return None, None

    min_dist = min(distances)
    avg_dist = sum(distances) / len(distances)

    return min_dist, avg_dist

    def ransac_yaw_estimation(self,ogm_1d, ego_extent_x, ego_extent_y, resolution=0.2,
                              grid_height=60, grid_width=60, yaw_history=deque(maxlen=5)):
        """
        Estimates yaw offset using RANSAC line fitting on occupancy grid data.
        Returns:
            - Estimated yaw difference (degrees)
            - Smoothed estimate using temporal history
        """
        ogm_2d = ogm_1d.reshape((grid_height, grid_width))

        # Define ROI - Focus on front-right quadrant (where boundary vehicle should be)
        center_row = grid_height // 2
        center_col = grid_width // 2

        # In ransac_yaw_estimation()
        vehicle_length_cells = int(ego_extent_x / resolution)   # Add buffer
        vehicle_width_cells = int(ego_extent_y / resolution)
        roi_rows = slice(center_row -  4, center_row + 4)
        roi_cols = slice(center_col + vehicle_width_cells, grid_width)

        roi = ogm_2d[roi_rows, roi_cols]

        # Get coordinates of occupied cells
        y, x = np.where(roi > 0.5)
        if len(x) < 10:  # Not enough points
            return None, np.mean(yaw_history) if yaw_history else 0.0

        # Convert to meters in vehicle coordinate system
        x_m = (x + roi_cols.start) * resolution - (grid_width / 2 * resolution)
        y_m = (y + roi_rows.start) * resolution - (grid_height / 2 * resolution)

        # RANSAC line fitting
        model = RANSACRegressor(residual_threshold=0.3, max_trials=100)
        model.fit(x_m.reshape(-1, 1), y_m)

        if not model.n_trials_ or model.estimator_ is None:
            return None, np.mean(yaw_history) if yaw_history else 0.0

        # Calculate angle from line parameters
        slope = model.estimator_.coef_[0]
        angle_rad = np.arctan(slope)
        angle_deg = np.degrees(angle_rad)

        """if model.estimator_ is not None:
            plt.scatter(x_m, y_m, c='r', s=2)
            x_range = np.array([x_m.min(), x_m.max()])
            plt.plot(x_range, model.predict(x_range.reshape(-1, 1)), 'b')
            plt.title("RANSAC Line Fit")
            plt.show()"""

        # Expected angle for perfect alignment (0° for straight boundary)
        yaw_diff = angle_deg - 0.0  # Adjust based on your scenario

        # Temporal smoothing
        yaw_history.append(yaw_diff)
        smoothed_yaw = np.mean(yaw_history)

        return yaw_diff, smoothed_yaw