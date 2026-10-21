"""
Utility functions for ASR700 Assignment.
Common helper functions used across components.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Optional, Any
import yaml
import json
from dataclasses import asdict


def world_to_grid(world_pos: Tuple[float, float],
                  grid_resolution: float,
                  world_origin: Tuple[float, float] = (0.0, 0.0)) -> Tuple[int, int]:
    """
    Convert world coordinates to grid indices.

    Args:
        world_pos: (x, y) in world coordinates (meters)
        grid_resolution: meters per grid cell
        world_origin: (x, y) of grid cell (0, 0) in world coordinates

    Returns:
        (row, col) grid indices
    """
    x, y = world_pos
    ox, oy = world_origin
    col = int((x - ox) / grid_resolution)
    row = int((y - oy) / grid_resolution)
    return row, col


def grid_to_world(grid_pos: Tuple[int, int],
                  grid_resolution: float,
                  world_origin: Tuple[float, float] = (0.0, 0.0)) -> Tuple[float, float]:
    """
    Convert grid indices to world coordinates (center of cell).

    Args:
        grid_pos: (row, col) grid indices
        grid_resolution: meters per grid cell
        world_origin: (x, y) of grid cell (0, 0) in world coordinates

    Returns:
        (x, y) world coordinates in meters
    """
    row, col = grid_pos
    ox, oy = world_origin
    x = ox + (col + 0.5) * grid_resolution
    y = oy + (row + 0.5) * grid_resolution
    return x, y


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def angle_difference(target: float, current: float) -> float:
    """Compute shortest angular difference from current to target."""
    diff = normalize_angle(target - current)
    return diff


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Euclidean distance between two 2D points."""
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])


def manhattan_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Manhattan distance between two 2D points."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def create_occupancy_grid(width: int, height: int,
                          default_value: float = 0.5) -> np.ndarray:
    """
    Create empty occupancy grid with log-odds representation.

    Args:
        width: Grid width in cells
        height: Grid height in cells
        default_value: Initial probability (0.5 = unknown)

    Returns:
        Log-odds grid (0 = free, 1 = occupied, 0.5 = unknown in probability space)
    """
    # Log-odds: L = log(p / (1-p))
    log_odds = np.log(default_value / (1 - default_value))
    return np.full((height, width), log_odds, dtype=np.float32)


def log_odds_to_probability(log_odds: np.ndarray) -> np.ndarray:
    """Convert log-odds grid to probability grid."""
    return 1.0 / (1.0 + np.exp(-log_odds))


def probability_to_log_odds(prob: np.ndarray) -> np.ndarray:
    """Convert probability grid to log-odds grid."""
    # Clamp to avoid log(0)
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    return np.log(prob / (1 - prob))


def update_occupancy_grid(grid: np.ndarray,
                          hits: List[Tuple[int, int]],
                          misses: List[Tuple[int, int]],
                          hit_odds: float = 0.85,
                          miss_odds: float = 0.4) -> np.ndarray:
    """
    Update occupancy grid with sensor measurements using log-odds.

    Args:
        grid: Current log-odds grid
        hits: List of (row, col) cells hit by sensor (occupied)
        misses: List of (row, col) cells along ray (free)
        hit_odds: Probability for occupied cells
        miss_odds: Probability for free cells

    Returns:
        Updated log-odds grid
    """
    hit_log_odds = np.log(hit_odds / (1 - hit_odds))
    miss_log_odds = np.log(miss_odds / (1 - miss_odds))

    for r, c in hits:
        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
            grid[r, c] += hit_log_odds

    for r, c in misses:
        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
            grid[r, c] += miss_log_odds

    # Clamp log-odds to reasonable range
    grid = np.clip(grid, -10, 10)
    return grid


def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    """
    Bresenham's line algorithm for grid ray tracing.

    Returns list of (row, col) cells along the line from (x0,y0) to (x1,y1).
    """
    cells = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    x, y = x0, y0
    while True:
        cells.append((y, x))  # (row, col)
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

    return cells


def lidar_to_grid_cells(lidar_ranges: np.ndarray,
                        angles: np.ndarray,
                        robot_pos: Tuple[float, float],
                        robot_heading: float,
                        grid_resolution: float,
                        grid_shape: Tuple[int, int],
                        world_origin: Tuple[float, float] = (0.0, 0.0),
                        max_range: float = 5.0) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Convert LiDAR scan to grid cells for occupancy grid update.

    Args:
        lidar_ranges: Array of range measurements
        angles: Array of corresponding angles (radians, relative to robot forward)
        robot_pos: (x, y) robot position in world coordinates
        robot_heading: Robot heading in radians
        grid_resolution: Grid cell size in meters
        grid_shape: (height, width) of grid
        world_origin: World coordinates of grid (0,0)
        max_range: Maximum sensor range

    Returns:
        (hits, misses) lists of grid cells
    """
    hits = []
    misses = []

    robot_row, robot_col = world_to_grid(robot_pos, grid_resolution, world_origin)
    grid_h, grid_w = grid_shape

    for range_val, angle in zip(lidar_ranges, angles):
        if range_val >= max_range or range_val <= 0:
            continue

        # Global angle of ray
        global_angle = robot_heading + angle

        # Endpoint in world coordinates
        end_x = robot_pos[0] + range_val * np.cos(global_angle)
        end_y = robot_pos[1] + range_val * np.sin(global_angle)
        end_row, end_col = world_to_grid((end_x, end_y), grid_resolution, world_origin)

        # Clamp to grid bounds
        end_row = np.clip(end_row, 0, grid_h - 1)
        end_col = np.clip(end_col, 0, grid_w - 1)

        # Ray trace from robot to endpoint
        ray_cells = bresenham_line(robot_col, robot_row, end_col, end_row)

        # All cells except last are misses (free space)
        for cell in ray_cells[:-1]:
            r, c = cell
            if 0 <= r < grid_h and 0 <= c < grid_w:
                misses.append((r, c))

        # Last cell is hit (occupied)
        if ray_cells:
            r, c = ray_cells[-1]
            if 0 <= r < grid_h and 0 <= c < grid_w:
                hits.append((r, c))

    return hits, misses


def save_yaml(data: dict, path: Path):
    """Save dictionary to YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_yaml(path: Path) -> dict:
    """Load dictionary from YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_json(data: dict, path: Path):
    """Save dictionary to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> dict:
    """Load dictionary from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_numpy(array: np.ndarray, path: Path):
    """Save numpy array to .npy file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def load_numpy(path: Path) -> np.ndarray:
    """Load numpy array from .npy file."""
    return np.load(path)


def compute_path_length(path: List[Tuple[float, float]]) -> float:
    """Compute total length of a path."""
    if len(path) < 2:
        return 0.0
    return sum(euclidean_distance(path[i], path[i+1]) for i in range(len(path)-1))


def smooth_path(path: List[Tuple[float, float]],
                weight_data: float = 0.5,
                weight_smooth: float = 0.1,
                tolerance: float = 1e-6) -> List[Tuple[float, float]]:
    """
    Smooth a path using gradient descent (similar to PID smoothing).

    Args:
        path: List of (x, y) waypoints
        weight_data: Weight for staying close to original path
        weight_smooth: Weight for smoothness
        tolerance: Convergence tolerance

    Returns:
        Smoothed path
    """
    if len(path) < 3:
        return path

    new_path = [list(p) for p in path]
    change = tolerance * 2

    while change > tolerance:
        change = 0.0
        for i in range(1, len(path) - 1):
            for j in range(2):  # x and y
                original = new_path[i][j]
                new_path[i][j] += weight_data * (path[i][j] - new_path[i][j])
                new_path[i][j] += weight_smooth * (new_path[i-1][j] + new_path[i+1][j] - 2 * new_path[i][j])
                change += abs(original - new_path[i][j])

    return [(p[0], p[1]) for p in new_path]


def interpolate_path(path: List[Tuple[float, float]],
                     step_size: float = 0.1) -> List[Tuple[float, float]]:
    """
    Interpolate path to have uniform step size.

    Args:
        path: List of (x, y) waypoints
        step_size: Desired distance between waypoints

    Returns:
        Interpolated path
    """
    if len(path) < 2:
        return path

    interpolated = [path[0]]
    for i in range(len(path) - 1):
        p1 = np.array(path[i])
        p2 = np.array(path[i+1])
        dist = np.linalg.norm(p2 - p1)
        if dist < step_size:
            continue
        num_steps = int(dist / step_size)
        for s in range(1, num_steps + 1):
            t = s / num_steps
            interp = p1 + t * (p2 - p1)
            interpolated.append((float(interp[0]), float(interp[1])))

    # Ensure goal is included
    if euclidean_distance(interpolated[-1], path[-1]) > 1e-6:
        interpolated.append(path[-1])

    return interpolated


def rotation_matrix_2d(angle: float) -> np.ndarray:
    """2D rotation matrix for given angle."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def transform_points(points: np.ndarray,
                     translation: Tuple[float, float],
                     rotation: float) -> np.ndarray:
    """
    Transform points by rotation and translation.

    Args:
        points: Nx2 array of points
        translation: (tx, ty)
        rotation: Angle in radians

    Returns:
        Transformed points
    """
    R = rotation_matrix_2d(rotation)
    T = np.array(translation)
    return (R @ points.T).T + T


class PIDController:
    """Simple PID controller implementation."""

    def __init__(self, kp: float, ki: float, kd: float,
                 output_limits: Tuple[float, float] = (-1.0, 1.0),
                 integral_limits: Tuple[float, float] = (-10.0, 10.0)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.integral_limits = integral_limits

        self.integral = 0.0
        self.previous_error = 0.0
        self.first_step = True

    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.previous_error = 0.0
        self.first_step = True

    def step(self, error: float, dt: float) -> float:
        """
        Compute PID output.

        Args:
            error: Current error (setpoint - measured)
            dt: Time step in seconds

        Returns:
            Control output
        """
        if dt <= 0:
            return 0.0

        # Proportional term
        p = self.kp * error

        # Integral term
        self.integral += error * dt
        self.integral = np.clip(self.integral, *self.integral_limits)
        i = self.ki * self.integral

        # Derivative term
        if self.first_step:
            d = 0.0
            self.first_step = False
        else:
            d = self.kd * (error - self.previous_error) / dt

        self.previous_error = error

        output = p + i + d
        return np.clip(output, *self.output_limits)


def compute_tracking_error(planned_path: List[Tuple[float, float]],
                           actual_trajectory: List[Tuple[float, float]]) -> dict:
    """
    Compute tracking error metrics between planned and actual paths.

    Args:
        planned_path: List of (x, y) planned waypoints
        actual_trajectory: List of (x, y) actual positions

    Returns:
        Dictionary of error metrics
    """
    if not planned_path or not actual_trajectory:
        return {"max_error": 0.0, "rmse": 0.0, "mean_error": 0.0}

    # For each actual point, find closest planned point
    planned = np.array(planned_path)
    actual = np.array(actual_trajectory)

    errors = []
    for a in actual:
        dists = np.linalg.norm(planned - a, axis=1)
        errors.append(np.min(dists))

    errors = np.array(errors)
    return {
        "max_error": float(np.max(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
    }


if __name__ == "__main__":
    # Quick tests
    print("Testing helpers...")

    # Test coordinate transforms
    grid_pos = world_to_grid((5.0, 5.0), 0.1, (0.0, 0.0))
    print(f"World (5,5) -> Grid {grid_pos}")

    world_pos = grid_to_world(grid_pos, 0.1, (0.0, 0.0))
    print(f"Grid {grid_pos} -> World {world_pos}")

    # Test angle normalization
    print(f"Normalize 3π: {normalize_angle(3*np.pi)}")
    print(f"Angle diff π/2 - (-π/2): {angle_difference(np.pi/2, -np.pi/2)}")

    # Test PID
    pid = PIDController(1.0, 0.1, 0.05)
    for i in range(10):
        out = pid.step(1.0, 0.1)
        print(f"PID step {i}: {out:.4f}")

    print("All tests passed!")