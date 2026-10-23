"""
Sensor Perception Pipeline for ASR700 Component A.

Implements LiDAR processing, camera image processing with OpenCV,
and sensor fusion for unified environment representation.
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
from python_analysis.utils.config import SensorConfig, EnvironmentConfig, RobotConfig
from python_analysis.utils.helpers import (lidar_to_grid_cells, world_to_grid,
                                           grid_to_world, create_occupancy_grid,
                                           log_odds_to_probability, update_occupancy_grid,
                                           normalize_angle)


class LandmarkColor(Enum):
    """Predefined landmark colors for detection."""
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    YELLOW = "yellow"


# HSV color ranges for landmark detection (H: 0-179, S: 0-255, V: 0-255)
COLOR_RANGES = {
    LandmarkColor.RED: [
        (np.array([0, 100, 100]), np.array([10, 255, 255])),
        (np.array([170, 100, 100]), np.array([179, 255, 255]))
    ],
    LandmarkColor.GREEN: [
        (np.array([40, 100, 100]), np.array([80, 255, 255]))
    ],
    LandmarkColor.BLUE: [
        (np.array([100, 100, 100]), np.array([130, 255, 255]))
    ],
    LandmarkColor.YELLOW: [
        (np.array([20, 100, 100]), np.array([35, 255, 255]))
    ]
}


@dataclass
class Landmark:
    """Detected landmark from camera."""
    color: LandmarkColor
    position: Tuple[float, float]  # World coordinates (x, y)
    pixel_position: Tuple[int, int]  # Image coordinates (u, v)
    area: float
    confidence: float
    timestamp: float = 0.0  # Detection timestamp


@dataclass
class SensorData:
    """Raw sensor data from Webots."""
    lidar_ranges: np.ndarray
    lidar_angles: np.ndarray
    camera_image: np.ndarray  # RGB image
    timestamp: float
    robot_pose: Tuple[float, float, float]  # x, y, theta


@dataclass
class FusedMap:
    """Fused environment representation."""
    occupancy_grid: np.ndarray  # Probability grid
    grid_resolution: float
    grid_origin: Tuple[float, float]
    landmarks: List[Landmark]
    obstacles_lidar: List[Tuple[float, float, float]]  # (x, y, radius) from LiDAR
    obstacles_camera: List[Tuple[float, float, float]]  # (x, y, radius) from camera
    robot_pose: Tuple[float, float, float]
    timestamp: float


class LiDARProcessor:
    """Process LiDAR range data into occupancy grid updates."""

    def __init__(self, config: SensorConfig = None, env_config: EnvironmentConfig = None):
        self.config = config or SensorConfig()
        self.env_config = env_config or EnvironmentConfig()

        # Angles are computed dynamically per scan to match actual beam count
        self.angles = None

    def process_scan(self, ranges: np.ndarray) -> np.ndarray:
        """
        Process raw LiDAR ranges.

        Args:
            ranges: Raw range measurements

        Returns:
            Filtered ranges (invalid readings set to max_range)
        """
        # Handle invalid readings
        filtered = np.copy(ranges)
        filtered[filtered <= self.config.lidar_min_range] = self.config.lidar_max_range
        filtered[filtered >= self.config.lidar_max_range] = self.config.lidar_max_range
        filtered[np.isnan(filtered)] = self.config.lidar_max_range
        filtered[np.isinf(filtered)] = self.config.lidar_max_range

        return filtered

    def ranges_to_points(self, ranges: np.ndarray,
                         robot_pose: Tuple[float, float, float]) -> np.ndarray:
        """
        Convert LiDAR ranges to world coordinate points.

        Args:
            ranges: Filtered range measurements
            robot_pose: (x, y, theta) robot pose

        Returns:
            Nx2 array of world points
        """
        rx, ry, rtheta = robot_pose

        # Compute angles to match actual number of beams in ranges
        n = len(ranges)
        angles = np.linspace(
            -np.deg2rad(self.config.lidar_fov / 2),
            np.deg2rad(self.config.lidar_fov / 2),
            n
        )

        # Filter valid ranges
        valid = (ranges > self.config.lidar_min_range) & (ranges < self.config.lidar_max_range)
        valid_ranges = ranges[valid]
        valid_angles = angles[valid]

        if len(valid_ranges) == 0:
            return np.empty((0, 2))

        # Global angles
        global_angles = rtheta + valid_angles

        # World coordinates
        xs = rx + valid_ranges * np.cos(global_angles)
        ys = ry + valid_ranges * np.sin(global_angles)

        return np.column_stack((xs, ys))

    def get_grid_cells(self, ranges: np.ndarray,
                       robot_pose: Tuple[float, float, float],
                       grid_shape: Tuple[int, int],
                       grid_resolution: float,
                       grid_origin: Tuple[float, float]) -> Tuple[List, List]:
        """
        Get grid cells for occupancy grid update.

        Returns:
            (hits, misses) lists of (row, col) grid cells
        """
        n = len(ranges)
        angles = np.linspace(
            -np.deg2rad(self.config.lidar_fov / 2),
            np.deg2rad(self.config.lidar_fov / 2),
            n
        )
        return lidar_to_grid_cells(
            ranges, angles,
            robot_pose[:2], robot_pose[2],
            grid_resolution, grid_shape,
            grid_origin, self.config.lidar_max_range
        )

    def cluster_points(self, ranges: np.ndarray,
                       robot_pose: Tuple[float, float, float],
                       distance_threshold: float = 0.2,
                       min_points: int = 3) -> List[Dict]:
        """
        Cluster LiDAR points into obstacles using Euclidean clustering.

        Args:
            ranges: Filtered range measurements
            robot_pose: (x, y, theta) robot pose
            distance_threshold: Max distance between points in same cluster
            min_points: Minimum points to form a cluster

        Returns:
            List of clusters with center, radius, point count
        """
        points = self.ranges_to_points(ranges, robot_pose)
        if len(points) == 0:
            return []

        clusters = []
        visited = np.zeros(len(points), dtype=bool)

        for i in range(len(points)):
            if visited[i]:
                continue

            # Start new cluster
            cluster_points = [points[i]]
            visited[i] = True
            queue = [i]

            while queue:
                idx = queue.pop(0)
                for j in range(len(points)):
                    if not visited[j]:
                        dist = np.linalg.norm(points[idx] - points[j])
                        if dist <= distance_threshold:
                            visited[j] = True
                            cluster_points.append(points[j])
                            queue.append(j)

            if len(cluster_points) >= min_points:
                cluster_points = np.array(cluster_points)
                center = np.mean(cluster_points, axis=0)
                # Radius as max distance from center
                radii = np.linalg.norm(cluster_points - center, axis=1)
                radius = np.max(radii)

                clusters.append({
                    'center': center,
                    'radius': radius,
                    'point_count': len(cluster_points),
                    'points': cluster_points
                })

        return clusters

    def detect_obstacles_from_clusters(self, ranges: np.ndarray,
                                        robot_pose: Tuple[float, float, float],
                                        distance_threshold: float = 0.2,
                                        min_points: int = 3) -> List[Tuple[float, float, float]]:
        """
        Detect obstacles as (x, y, radius) from LiDAR clusters.

        Returns:
            List of (x, y, radius) in world coordinates
        """
        clusters = self.cluster_points(ranges, robot_pose, distance_threshold, min_points)
        return [(c['center'][0], c['center'][1], c['radius']) for c in clusters]


class CameraProcessor:
    """Process camera images for landmark detection."""

    def __init__(self, config: SensorConfig = None):
        self.config = config or SensorConfig()
        self.camera_matrix = None
        self.dist_coeffs = None

    def set_calibration(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        """Set camera calibration parameters."""
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def undistort(self, image: np.ndarray) -> np.ndarray:
        """Undistort image using calibration."""
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)
        return image

    def detect_landmarks(self, image: np.ndarray,
                         robot_pose: Tuple[float, float, float],
                         camera_fov: float = 60.0,
                         timestamp: float = 0.0) -> List[Landmark]:
        """
        Detect colored landmarks in image.

        Args:
            image: RGB image (H, W, 3)
            robot_pose: (x, y, theta) robot pose
            camera_fov: Camera horizontal FOV in degrees

        Returns:
            List of detected landmarks
        """
        landmarks = []
        h, w = image.shape[:2]

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Optional: undistort
        # hsv = self.undistort(hsv)

        rx, ry, rtheta = robot_pose

        for color, ranges in COLOR_RANGES.items():
            # Create mask for this color
            mask = np.zeros((h, w), dtype=np.uint8)
            for lower, upper in ranges:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 100:  # Minimum area threshold
                    continue

                # Get centroid
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue

                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Estimate distance from apparent size
                # Assuming known landmark size (e.g., 0.1m diameter cylinder)
                # This is a simplified estimation
                equivalent_diameter = np.sqrt(4 * area / np.pi)
                focal_length_px = (w / 2) / np.tan(np.deg2rad(camera_fov / 2))
                estimated_distance = (0.1 * focal_length_px) / equivalent_diameter if equivalent_diameter > 0 else 5.0

                # Clamp distance
                estimated_distance = np.clip(estimated_distance, 0.2, 5.0)

                # Calculate bearing from image center
                bearing = (cx - w / 2) / w * np.deg2rad(camera_fov)

                # World position
                global_bearing = rtheta + bearing
                lx = rx + estimated_distance * np.cos(global_bearing)
                ly = ry + estimated_distance * np.sin(global_bearing)

                # Confidence based on area and circularity
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                confidence = min(1.0, (area / 5000) * circularity)

                landmark = Landmark(
                    color=color,
                    position=(lx, ly),
                    pixel_position=(cx, cy),
                    area=area,
                    confidence=confidence,
                    timestamp=timestamp
                )
                landmarks.append(landmark)

        return landmarks

    def detect_edges(self, image: np.ndarray,
                     low_threshold: int = 50,
                     high_threshold: int = 150) -> np.ndarray:
        """
        Detect edges using Canny edge detector.

        Args:
            image: RGB or grayscale image
            low_threshold: Canny low threshold
            high_threshold: Canny high threshold

        Returns:
            Binary edge map
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Gaussian blur for noise reduction
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge detection
        edges = cv2.Canny(blurred, low_threshold, high_threshold)

        return edges

    def detect_obstacles(self, image: np.ndarray,
                         robot_pose: Tuple[float, float, float],
                         depth_image: np.ndarray = None,
                         camera_fov: float = 60.0) -> List[Tuple[float, float, float]]:
        """
        Detect obstacles from camera using edge detection and optional depth.

        Args:
            image: RGB image
            robot_pose: (x, y, theta) robot pose
            depth_image: Optional depth image (H, W) in meters
            camera_fov: Camera horizontal FOV in degrees

        Returns:
            List of (x, y, radius) obstacle positions in world coordinates
        """
        obstacles = []
        h, w = image.shape[:2]
        rx, ry, rtheta = robot_pose

        # Detect edges
        edges = self.detect_edges(image)

        # Find contours of edge clusters
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 200:  # Minimum area threshold
                continue

            # Get bounding box
            x, y, bw, bh = cv2.boundingRect(contour)
            cx, cy = x + bw // 2, y + bh // 2

            # Estimate distance
            if depth_image is not None and 0 <= cy < h and 0 <= cx < w:
                distance = depth_image[cy, cx]
                if distance <= 0 or distance > 5.0:
                    continue
            else:
                # Estimate from apparent size (rough approximation)
                equivalent_diameter = np.sqrt(4 * area / np.pi)
                focal_length_px = (w / 2) / np.tan(np.deg2rad(camera_fov / 2))
                distance = (0.2 * focal_length_px) / equivalent_diameter if equivalent_diameter > 0 else 5.0
                distance = np.clip(distance, 0.3, 5.0)

            # Calculate bearing
            bearing = (cx - w / 2) / w * np.deg2rad(camera_fov)
            global_bearing = rtheta + bearing

            # World position
            ox = rx + distance * np.cos(global_bearing)
            oy = ry + distance * np.sin(global_bearing)

            # Estimate radius from bounding box
            radius = max(bw, bh) / w * distance * np.tan(np.deg2rad(camera_fov / 2)) * 2
            radius = np.clip(radius, 0.1, 1.0)

            obstacles.append((ox, oy, radius))

        return obstacles


class SensorFusion:
    """
    Fuse LiDAR and camera data into unified environment representation.
    """

    def __init__(self, sensor_config: SensorConfig = None,
                 env_config: EnvironmentConfig = None,
                 robot_config: RobotConfig = None):
        self.sensor_config = sensor_config or SensorConfig()
        self.env_config = env_config or EnvironmentConfig()
        self.robot_config = robot_config or RobotConfig()

        # Grid setup
        self.grid_resolution = env_config.grid_resolution
        self.grid_origin = (0.0, 0.0)
        grid_w = int(env_config.world_size[0] / self.grid_resolution)
        grid_h = int(env_config.world_size[1] / self.grid_resolution)
        self.grid_shape = (grid_h, grid_w)

        # Initialize occupancy grid (log-odds)
        self.occupancy_grid = create_occupancy_grid(grid_w, grid_h, 0.5)

        # Processors
        self.lidar_processor = LiDARProcessor(sensor_config, env_config)
        self.camera_processor = CameraProcessor(sensor_config)

        # Landmarks memory (persistent)
        self.known_landmarks: Dict[str, Landmark] = {}
        self.landmark_timeout = 5.0  # seconds

    def update(self, sensor_data: SensorData) -> FusedMap:
        """
        Fuse new sensor data into map.

        Args:
            sensor_data: Raw sensor data

        Returns:
            Updated fused map
        """
        robot_pose = sensor_data.robot_pose

        # Process LiDAR
        filtered_ranges = self.lidar_processor.process_scan(sensor_data.lidar_ranges)

        # Update occupancy grid
        hits, misses = self.lidar_processor.get_grid_cells(
            filtered_ranges, robot_pose, self.grid_shape,
            self.grid_resolution, self.grid_origin
        )
        self.occupancy_grid = update_occupancy_grid(
            self.occupancy_grid, hits, misses
        )

        # Detect obstacles from LiDAR clusters
        obstacles_lidar = self.lidar_processor.detect_obstacles_from_clusters(
            filtered_ranges, robot_pose
        )

        # Process camera
        landmarks = self.camera_processor.detect_landmarks(
            sensor_data.camera_image, robot_pose, timestamp=sensor_data.timestamp
        )

        # Detect obstacles from camera
        depth_image = getattr(sensor_data, 'depth_image', None)
        obstacles_camera = self.camera_processor.detect_obstacles(
            sensor_data.camera_image, robot_pose, depth_image=depth_image
        )

        # Update landmark memory
        self._update_landmark_memory(landmarks, sensor_data.timestamp)

        # Create fused map
        fused = FusedMap(
            occupancy_grid=log_odds_to_probability(self.occupancy_grid),
            grid_resolution=self.grid_resolution,
            grid_origin=self.grid_origin,
            landmarks=list(self.known_landmarks.values()),
            obstacles_lidar=obstacles_lidar,
            obstacles_camera=obstacles_camera,
            robot_pose=robot_pose,
            timestamp=sensor_data.timestamp
        )

        # Store for later access
        self._last_fused_map = fused

        return fused

    def _update_landmark_memory(self, new_landmarks: List[Landmark], timestamp: float):
        """Update persistent landmark memory with new detections."""
        # Associate new landmarks with known ones
        for new_lm in new_landmarks:
            # Simple association by color and proximity
            key = f"{new_lm.color.value}_{int(new_lm.position[0]*10)}_{int(new_lm.position[1]*10)}"

            if key in self.known_landmarks:
                # Update existing
                old = self.known_landmarks[key]
                # Weighted average position
                alpha = 0.3
                new_pos = (
                    (1 - alpha) * old.position[0] + alpha * new_lm.position[0],
                    (1 - alpha) * old.position[1] + alpha * new_lm.position[1]
                )
                new_conf = max(old.confidence, new_lm.confidence)
                self.known_landmarks[key] = Landmark(
                    color=new_lm.color,
                    position=new_pos,
                    pixel_position=new_lm.pixel_position,
                    area=new_lm.area,
                    confidence=new_conf
                )
            else:
                # New landmark
                self.known_landmarks[key] = new_lm

        # Remove old landmarks
        to_remove = []
        for key, lm in self.known_landmarks.items():
            # Simple timeout (would need more robust tracking in practice)
            if timestamp - lm.timestamp > self.landmark_timeout:
                to_remove.append(key)

        for key in to_remove:
            del self.known_landmarks[key]

    def get_occupancy_grid(self) -> np.ndarray:
        """Get current occupancy grid as probabilities."""
        return log_odds_to_probability(self.occupancy_grid)

    def get_log_odds_grid(self) -> np.ndarray:
        """Get current occupancy grid as log-odds."""
        return self.occupancy_grid.copy()

    def reset(self):
        """Reset fusion state."""
        self.occupancy_grid = create_occupancy_grid(
            self.grid_shape[1], self.grid_shape[0], 0.5)
        self.known_landmarks.clear()


class PerceptionPipeline:
    """
    Complete perception pipeline for Component A.

    Integrates LiDAR processing, camera processing, and sensor fusion.
    """

    def __init__(self, sensor_config: SensorConfig = None,
                 env_config: EnvironmentConfig = None,
                 robot_config: RobotConfig = None):
        self.sensor_config = sensor_config or SensorConfig()
        self.env_config = env_config or EnvironmentConfig()
        self.robot_config = robot_config or RobotConfig()

        self.fusion = SensorFusion(sensor_config, env_config, robot_config)
        self.lidar_processor = LiDARProcessor(sensor_config, env_config)
        self.camera_processor = CameraProcessor(sensor_config)

    def process(self, lidar_ranges: np.ndarray,
                camera_image: np.ndarray,
                robot_pose: Tuple[float, float, float],
                timestamp: float = 0.0) -> FusedMap:
        """
        Process sensor data through full pipeline.

        Args:
            lidar_ranges: Raw LiDAR ranges
            camera_image: RGB camera image
            robot_pose: (x, y, theta) robot pose
            timestamp: Current time

        Returns:
            Fused environment map
        """
        sensor_data = SensorData(
            lidar_ranges=lidar_ranges,
            lidar_angles=self.lidar_processor.angles,
            camera_image=camera_image,
            timestamp=timestamp,
            robot_pose=robot_pose
        )

        return self.fusion.update(sensor_data)

    def get_obstacle_map(self) -> np.ndarray:
        """Get binary obstacle map for planning."""
        prob_grid = self.fusion.get_occupancy_grid()
        return (prob_grid >= 0.5).astype(np.uint8)

    def get_landmarks(self) -> List[Landmark]:
        """Get current landmarks."""
        return self.fusion.known_landmarks.values()

    def get_obstacles_lidar(self) -> List[Tuple[float, float, float]]:
        """Get obstacles detected from LiDAR."""
        if hasattr(self.fusion, '_last_fused_map'):
            return self.fusion._last_fused_map.obstacles_lidar
        return []

    def get_obstacles_camera(self) -> List[Tuple[float, float, float]]:
        """Get obstacles detected from camera."""
        if hasattr(self.fusion, '_last_fused_map'):
            return self.fusion._last_fused_map.obstacles_camera
        return []

    def get_all_obstacles(self) -> List[Tuple[float, float, float]]:
        """Get combined obstacles from both sensors."""
        lidar_obs = self.get_obstacles_lidar()
        cam_obs = self.get_obstacles_camera()
        return lidar_obs + cam_obs

    def reset(self):
        """Reset pipeline state."""
        self.fusion.reset()


def create_occupancy_grid_from_lidar(lidar_ranges: np.ndarray,
                                     lidar_angles: np.ndarray,
                                     robot_pose: Tuple[float, float, float],
                                     grid_resolution: float = 0.05,
                                     world_size: Tuple[float, float] = (10.0, 10.0),
                                     max_range: float = 5.0) -> np.ndarray:
    """
    Create occupancy grid from single LiDAR scan.

    Args:
        lidar_ranges: Range measurements
        lidar_angles: Corresponding angles
        robot_pose: Robot (x, y, theta)
        grid_resolution: Grid cell size
        world_size: World dimensions
        max_range: Maximum sensor range

    Returns:
        Probability occupancy grid
    """
    grid_w = int(world_size[0] / grid_resolution)
    grid_h = int(world_size[1] / grid_resolution)
    grid_shape = (grid_h, grid_w)
    grid_origin = (0.0, 0.0)

    # Create log-odds grid
    grid = create_occupancy_grid(grid_w, grid_h, 0.5)

    # Get grid cells
    hits, misses = lidar_to_grid_cells(
        lidar_ranges, lidar_angles,
        robot_pose[:2], robot_pose[2],
        grid_resolution, grid_shape,
        grid_origin, max_range
    )

    # Update grid
    grid = update_occupancy_grid(grid, hits, misses)

    return log_odds_to_probability(grid)


def visualize_perception(fused_map: FusedMap,
                          save_path: str = None,
                          show: bool = True):
    """
    Visualize fused perception data.

    Args:
        fused_map: FusedMap from perception pipeline
        save_path: Optional path to save figure
        show: Whether to display the plot
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Occupancy grid with obstacles
    ax = axes[0]
    im = ax.imshow(fused_map.occupancy_grid, origin='lower', cmap='RdYlGn_r',
                   vmin=0, vmax=1,
                   extent=[fused_map.grid_origin[0],
                           fused_map.grid_origin[0] + fused_map.occupancy_grid.shape[1] * fused_map.grid_resolution,
                           fused_map.grid_origin[1],
                           fused_map.grid_origin[1] + fused_map.occupancy_grid.shape[0] * fused_map.grid_resolution],
                   alpha=0.7)

    # Robot position
    rx, ry, rtheta = fused_map.robot_pose
    ax.scatter(rx, ry, c='red', s=100, marker='^', label='Robot', zorder=5)
    # Robot heading
    arrow_len = 0.5
    ax.arrow(rx, ry, arrow_len * np.cos(rtheta), arrow_len * np.sin(rtheta),
             head_width=0.2, head_length=0.1, fc='red', ec='red', zorder=5)

    # LiDAR obstacles
    for ox, oy, rad in fused_map.obstacles_lidar:
        circle = plt.Circle((ox, oy), rad, color='blue', fill=True, alpha=0.3, label='LiDAR Obstacle')
        ax.add_patch(circle)

    # Camera obstacles
    for ox, oy, rad in fused_map.obstacles_camera:
        circle = plt.Circle((ox, oy), rad, color='orange', fill=True, alpha=0.3, label='Camera Obstacle')
        ax.add_patch(circle)

    # Landmarks
    for lm in fused_map.landmarks:
        color_map = {'red': 'red', 'green': 'green', 'blue': 'blue', 'yellow': 'gold'}
        c = color_map.get(lm.color.value, 'black')
        ax.scatter(lm.position[0], lm.position[1], c=c, s=150, marker='*',
                   edgecolors='black', linewidth=1, zorder=5, label=f'{lm.color.value} landmark')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Fused Perception: Occupancy Grid + Obstacles + Landmarks')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='Occupancy Probability')

    # Occupancy grid only (clean)
    ax = axes[1]
    im = ax.imshow(fused_map.occupancy_grid, origin='lower', cmap='RdYlGn_r',
                   vmin=0, vmax=1,
                   extent=[fused_map.grid_origin[0],
                           fused_map.grid_origin[0] + fused_map.occupancy_grid.shape[1] * fused_map.grid_resolution,
                           fused_map.grid_origin[1],
                           fused_map.grid_origin[1] + fused_map.occupancy_grid.shape[0] * fused_map.grid_resolution])
    ax.scatter(rx, ry, c='red', s=100, marker='^', zorder=5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Occupancy Grid (Probability)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='Occupancy Probability')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

    return fig


def visualize_lidar_scan(ranges: np.ndarray,
                         angles: np.ndarray,
                         robot_pose: Tuple[float, float, float],
                         clusters: List[Dict] = None,
                         save_path: str = None,
                         show: bool = True):
    """
    Visualize LiDAR scan with optional clusters.

    Args:
        ranges: LiDAR range measurements
        angles: Corresponding angles
        robot_pose: (x, y, theta) robot pose
        clusters: Optional list of clusters from cluster_points()
        save_path: Optional path to save figure
        show: Whether to display the plot
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    rx, ry, rtheta = robot_pose

    # Polar plot
    ax = axes[0]
    ax = plt.subplot(121, projection='polar')
    ax.plot(angles, ranges, 'b-', linewidth=0.5)
    ax.fill(angles, ranges, alpha=0.2, color='blue')
    ax.set_title('LiDAR Scan (Polar)')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # Cartesian with clusters
    ax = axes[1]
    valid = ranges < 5.0
    if np.any(valid):
        xs = rx + ranges[valid] * np.cos(rtheta + angles[valid])
        ys = ry + ranges[valid] * np.sin(rtheta + angles[valid])
        ax.scatter(xs, ys, c='blue', s=5, alpha=0.5, label='LiDAR Points')

    ax.scatter(rx, ry, c='red', s=100, marker='^', label='Robot', zorder=5)

    if clusters:
        for i, cluster in enumerate(clusters):
            center = cluster['center']
            radius = cluster['radius']
            points = cluster['points']
            ax.scatter(points[:, 0], points[:, 1], s=10, alpha=0.7, label=f'Cluster {i+1}')
            circle = plt.Circle(center, radius, color='red', fill=False, linewidth=2)
            ax.add_patch(circle)
            ax.scatter(center[0], center[1], c='red', s=50, marker='x', zorder=5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('LiDAR Points + Clusters (Cartesian)')
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

    return fig


if __name__ == "__main__":
    # Test perception components
    from python_analysis.utils.config import SensorConfig, EnvironmentConfig

    sensor_config = SensorConfig()
    env_config = EnvironmentConfig()

    # Test LiDAR processor
    lidar = LiDARProcessor(sensor_config, env_config)
    print(f"LiDAR beams: {len(lidar.angles)}")
    print(f"Angle range: {np.rad2deg(lidar.angles[0]):.1f} to {np.rad2deg(lidar.angles[-1]):.1f} deg")

    # Test Camera processor
    camera = CameraProcessor(sensor_config)
    print(f"Camera: {sensor_config.camera_width}x{sensor_config.camera_height}")

    # Test sensor fusion
    fusion = SensorFusion(sensor_config, env_config)
    print(f"Grid: {fusion.grid_shape[1]}x{fusion.grid_shape[0]} cells")
    print(f"Resolution: {fusion.grid_resolution} m/cell")

    # Test landmark color ranges
    for color, ranges in COLOR_RANGES.items():
        print(f"{color.value}: {len(ranges)} HSV range(s)")

    print("Perception module tests passed!")