"""
Main Robot Controller for Component A.

Integrates perception, planning, and control for autonomous navigation
in Webots simulation.
"""

import sys
import os
import time
import json
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field, asdict

# Add python_analysis to path for imports
controller_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(controller_dir)))
sys.path.insert(0, project_root)

from python_analysis.utils.config import (SensorConfig, EnvironmentConfig,
                                           RobotConfig, PIDConfig, PathPlanningConfig)
from python_analysis.utils.helpers import (normalize_angle, angle_difference,
                                           euclidean_distance, interpolate_path,
                                           compute_tracking_error, compute_path_length)
from webots_project.controllers.robot_controller_a.perception import PerceptionPipeline
from webots_project.controllers.robot_controller_a.planning import (plan_path, PlannerType,
                                                                     OccupancyGridMap)
from webots_project.controllers.robot_controller_a.control import (PathTrackingController, PIDConfig)
from webots_project.controllers.robot_controller_a.kinematics import (DifferentialDriveKinematics,
                                                                       RobotState)

# Webots imports
try:
    from controller import Robot, Lidar, Camera, Motor, GPS, InertialUnit, Supervisor
except ImportError:
    # Mock for testing outside Webots
    class Robot:
        def __init__(self):
            self.timestep = 32
        def step(self, timestep):
            return -1
        def getBasicTimeStep(self):
            return 32

    class Lidar:
        def __init__(self, name): pass
        def enable(self, timestep): pass
        def getRangeImage(self): return []

    class Camera:
        def __init__(self, name): pass
        def enable(self, timestep): pass
        def getImageArray(self): return None

    class Motor:
        def __init__(self, name): pass
        def setPosition(self, pos): pass
        def setVelocity(self, vel): pass

    class GPS:
        def __init__(self, name): pass
        def enable(self, timestep): pass
        def getValues(self): return [0, 0, 0]

    class InertialUnit:
        def __init__(self, name): pass
        def enable(self, timestep): pass
        def getRollPitchYaw(self): return [0, 0, 0]


@dataclass
class NavigationMetrics:
    """Metrics for navigation performance evaluation."""
    planned_path_length: float = 0.0
    actual_path_length: float = 0.0
    path_efficiency: float = 0.0
    time_to_goal: float = 0.0
    max_tracking_error: float = 0.0
    rmse_tracking_error: float = 0.0
    mean_tracking_error: float = 0.0
    std_tracking_error: float = 0.0
    num_replans: int = 0
    goal_reached: bool = False
    collisions: int = 0
    replan_reasons: List[str] = field(default_factory=list)
    control_effort: float = 0.0  # Sum of |v| + |omega|


class RobotControllerA:
    """
    Main controller for Component A: Single Robot Autonomous Navigation.

    Integrates:
    - Sensor perception (LiDAR + Camera fusion)
    - Motion planning (Dijkstra + A*)
    - Control (PID path tracking)
    """

    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        # Configuration
        self.sensor_config = SensorConfig()
        self.env_config = EnvironmentConfig()
        self.robot_config = RobotConfig()
        self.pid_config = PIDConfig()
        self.planning_config = PathPlanningConfig()

        # Initialize devices
        self._init_devices()

        # Initialize subsystems
        self.perception = PerceptionPipeline(
            self.sensor_config, self.env_config, self.robot_config)
        self.kinematics = DifferentialDriveKinematics(self.robot_config)

        # Planning
        self.occupancy_map: Optional[OccupancyGridMap] = None
        self.planned_path: List[Tuple[float, float]] = []
        self.path_interpolated: List[Tuple[float, float]] = []
        self.current_waypoint_idx = 0

        # Control
        self.controller = PathTrackingController(
            self.kinematics, self.pid_config,
            waypoint_tolerance=0.15,
            lookahead_distance=0.3
        )

        # State
        self.state = RobotState()
        self.goal: Optional[Tuple[float, float]] = None
        self.start_time = time.time()
        self.last_time = self.start_time

        # Logging
        self.trajectory_log: List[Tuple[float, float, float]] = []  # (x, y, t)
        self.planned_path_log: List[Tuple[float, float]] = []
        self.control_log: List[Dict[str, Any]] = []
        self.replan_log: List[Dict[str, Any]] = []

        # Planning state
        self.path_planned = False
        self.replan_counter = 0
        self.replan_interval = 50  # Replan every N steps
        self.replan_triggered_by_error = False

        # Metrics
        self.metrics = NavigationMetrics()
        self.last_replan_time = 0.0
        self.min_replan_interval = 2.0  # Minimum seconds between replans

        # Safety
        self.emergency_stop = False
        self.collision_threshold = 0.2  # meters
        self.max_tracking_error = 0.5  # meters - trigger replan if exceeded

        # Dynamic obstacles tracking
        self.dynamic_obstacles: List[Tuple[float, float, float]] = []  # (x, y, radius)

        # Data export path
        self.export_dir = os.path.join(project_root, "data", "trajectories")
        os.makedirs(self.export_dir, exist_ok=True)

    def _init_devices(self):
        """Initialize Webots devices."""
        # LiDAR
        self.lidar = self.robot.getDevice('lidar')
        if self.lidar:
            self.lidar.enable(self.timestep)
            try:
                self.lidar.enablePointCloud()
            except Exception:
                pass

        # Camera
        self.camera = self.robot.getDevice('camera')
        if self.camera:
            self.camera.enable(self.timestep)
            try:
                self.camera.recognitionEnable(self.timestep)
            except Exception:
                pass  # e-puck camera has no Recognition node

        # Motors
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        if self.left_motor and self.right_motor:
            self.left_motor.setPosition(float('inf'))
            self.right_motor.setPosition(float('inf'))
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)

        # GPS
        self.gps = self.robot.getDevice('gps')
        if self.gps:
            self.gps.enable(self.timestep)

        # Inertial Unit
        self.imu = self.robot.getDevice('inertial unit')
        if self.imu:
            self.imu.enable(self.timestep)

        # Distance sensors (for collision detection)
        self.distance_sensors = []
        for i in range(8):
            ds = self.robot.getDevice(f'ps{i}')
            if ds:
                ds.enable(self.timestep)
                self.distance_sensors.append(ds)

        print("Devices initialized")

    def _read_sensors(self) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float], List[float]]:
        """Read all sensors and return processed data."""
        # LiDAR
        lidar_ranges = np.array([])
        if self.lidar:
            raw_ranges = self.lidar.getRangeImage()
            if raw_ranges:
                lidar_ranges = np.array(raw_ranges)
                # Webots returns ranges in reverse order sometimes
                lidar_ranges = lidar_ranges[::-1]

        # Camera
        camera_image = np.zeros((self.sensor_config.camera_height,
                                 self.sensor_config.camera_width, 3), dtype=np.uint8)
        if self.camera:
            img_data = self.camera.getImageArray()
            if img_data:
                camera_image = np.array(img_data, dtype=np.uint8)

        # Pose from GPS + IMU
        pose = (0.0, 0.0, 0.0)
        if self.gps and self.imu:
            gps_vals = self.gps.getValues()
            imu_vals = self.imu.getRollPitchYaw()
            pose = (gps_vals[0], gps_vals[1], imu_vals[2])  # x, y, yaw

        # Distance sensors (for collision detection)
        ds_values = []
        for ds in self.distance_sensors:
            ds_values.append(ds.getValue())

        return lidar_ranges, camera_image, pose, ds_values

    def _check_collision(self, ds_values: List[float]) -> bool:
        """Check for imminent collision using distance sensors."""
        # Webots e-puck proximity sensors: higher values = closer (range ~0-4000)
        # Threshold for ~3cm detection: ~2000 (max range ~5cm at ~4000)
        for val in ds_values:
            if val > 2000:  # High threshold for imminent collision only
                return True
        return False

    def _update_dynamic_obstacles(self, fused_map):
        """Track dynamic obstacles from perception."""
        # Combine LiDAR and camera obstacles
        all_obstacles = fused_map.obstacles_lidar + fused_map.obstacles_camera

        # Simple tracking: match with previous obstacles by proximity
        matched = set()
        new_obstacles = []

        for obs in all_obstacles:
            ox, oy, rad = obs
            best_match = None
            best_dist = float('inf')

            for i, (px, py, prad) in enumerate(self.dynamic_obstacles):
                if i in matched:
                    continue
                dist = euclidean_distance((ox, oy), (px, py))
                if dist < 0.5 and dist < best_dist:  # Match threshold
                    best_dist = dist
                    best_match = i

            if best_match is not None:
                # Update existing obstacle (moving average)
                matched.add(best_match)
                px, py, prad = self.dynamic_obstacles[best_match]
                alpha = 0.3
                new_x = (1 - alpha) * px + alpha * ox
                new_y = (1 - alpha) * py + alpha * oy
                new_rad = max(prad, rad)
                new_obstacles.append((new_x, new_y, new_rad))
            else:
                # New obstacle
                new_obstacles.append(obs)

        # Keep unmatched old obstacles for a short time (could add timeout)
        for i, obs in enumerate(self.dynamic_obstacles):
            if i not in matched:
                new_obstacles.append(obs)

        self.dynamic_obstacles = new_obstacles[:20]  # Limit number

    def _add_obstacles_to_map(self, fused_map):
        """Add perceived obstacles to occupancy map."""
        if self.occupancy_map is None:
            return

        all_obstacles = fused_map.obstacles_lidar + fused_map.obstacles_camera

        for ox, oy, rad in all_obstacles:
            # Convert obstacle to grid cells
            cx, cy = self.occupancy_map.world_to_grid((ox, oy))
            r_cells = int(rad / self.occupancy_map.resolution) + 1

            for dr in range(-r_cells, r_cells + 1):
                for dc in range(-r_cells, r_cells + 1):
                    if dr**2 + dc**2 <= r_cells**2:
                        nr, nc = cy + dr, cx + dc
                        if self.occupancy_map.is_valid(nr, nc):
                            # Mark as occupied with high probability
                            self.occupancy_map.grid[nr, nc] = 0.9
                            self.occupancy_map.obstacle_map[nr, nc] = 1

    def _update_state(self, pose: Tuple[float, float, float]):
        """Update internal state estimate."""
        self.state.x = pose[0]
        self.state.y = pose[1]
        self.state.theta = normalize_angle(pose[2])

        # Log trajectory
        self.trajectory_log.append((self.state.x, self.state.y))

    def _check_collision(self, ds_values: List[float]) -> bool:
        """Check for imminent collision using distance sensors."""
        # Webots e-puck proximity sensors: higher values = closer (range ~0-4000)
        # Threshold for ~3cm detection: ~2000 (max range ~5cm at ~4000)
        for val in ds_values:
            if val > 2000:  # High threshold for imminent collision only
                return True
        return False

    def _update_dynamic_obstacles(self, fused_map):
        """Track dynamic obstacles from perception."""
        # Combine LiDAR and camera obstacles
        all_obstacles = fused_map.obstacles_lidar + fused_map.obstacles_camera

        # Simple tracking: match with previous obstacles by proximity
        matched = set()
        new_obstacles = []

        for obs in all_obstacles:
            ox, oy, rad = obs
            best_match = None
            best_dist = float('inf')

            for i, (px, py, prad) in enumerate(self.dynamic_obstacles):
                if i in matched:
                    continue
                dist = euclidean_distance((ox, oy), (px, py))
                if dist < 0.5 and dist < best_dist:  # Match threshold
                    best_dist = dist
                    best_match = i

            if best_match is not None:
                # Update existing obstacle (moving average)
                matched.add(best_match)
                px, py, prad = self.dynamic_obstacles[best_match]
                alpha = 0.3
                new_x = (1 - alpha) * px + alpha * ox
                new_y = (1 - alpha) * py + alpha * oy
                new_rad = max(prad, rad)
                new_obstacles.append((new_x, new_y, new_rad))
            else:
                # New obstacle
                new_obstacles.append(obs)

        # Keep unmatched old obstacles for a short time (could add timeout)
        for i, obs in enumerate(self.dynamic_obstacles):
            if i not in matched:
                new_obstacles.append(obs)

        self.dynamic_obstacles = new_obstacles[:20]  # Limit number

    def _add_obstacles_to_map(self, fused_map):
        """Add perceived obstacles to occupancy map."""
        if self.occupancy_map is None:
            return

        all_obstacles = fused_map.obstacles_lidar + fused_map.obstacles_camera

        for ox, oy, rad in all_obstacles:
            # Convert obstacle to grid cells
            cx, cy = self.occupancy_map.world_to_grid((ox, oy))
            r_cells = int(rad / self.occupancy_map.resolution) + 1

            for dr in range(-r_cells, r_cells + 1):
                for dc in range(-r_cells, r_cells + 1):
                    if dr**2 + dc**2 <= r_cells**2:
                        nr, nc = cy + dr, cx + dc
                        if self.occupancy_map.is_valid(nr, nc):
                            # Mark as occupied with high probability
                            self.occupancy_map.grid[nr, nc] = 0.9
                            self.occupancy_map.obstacle_map[nr, nc] = 1

    def _initialize_planning(self):
        """Initialize occupancy map and plan initial path."""
        # Create occupancy map from environment config
        grid_w = int(self.env_config.world_size[0] / self.env_config.grid_resolution)
        grid_h = int(self.env_config.world_size[1] / self.env_config.grid_resolution)

        # Initialize with empty grid (will be updated from perception)
        empty_grid = np.zeros((grid_h, grid_w), dtype=np.float32)
        self.occupancy_map = OccupancyGridMap(
            empty_grid,
            self.env_config.grid_resolution,
            (0.0, 0.0),
            occupancy_threshold=0.5
        )

        # Set goal (default: opposite corner)
        self.goal = self.env_config.goal_position
        self.start_time = time.time()
        self.last_replan_time = self.start_time

        # Plan initial path
        self._replan()

    def _replan(self, reason: str = "periodic"):
        """Replan path using current occupancy map."""
        if self.occupancy_map is None or self.goal is None:
            return

        current_time = time.time() - self.start_time

        # Respect minimum replan interval
        if current_time - self.last_replan_time < self.min_replan_interval:
            return

        start = (self.state.x, self.state.y)

        # Use A* for planning
        path, stats = plan_path(
            self.occupancy_map, start, self.goal,
            PlannerType.ASTAR,
            diagonal=True,
            heuristic_type="euclidean"
        )

        if stats.path_found:
            old_path_len = compute_path_length(self.planned_path) if self.planned_path else 0
            self.planned_path = path
            # Interpolate for smoother tracking
            self.path_interpolated = interpolate_path(path, step_size=0.1)
            self.controller.set_path(self.path_interpolated)
            self.current_waypoint_idx = 0
            self.path_planned = True

            # Log for comparison
            self.planned_path_log = path.copy()

            self.metrics.num_replans += 1
            self.metrics.replan_reasons.append(f"{reason} (t={current_time:.1f}s)")
            self.last_replan_time = current_time

            print(f"Path replanned ({reason}): {len(path)} waypoints, length={stats.path_length:.2f}m, "
                  f"nodes={stats.nodes_explored}, time={stats.computation_time_ms:.1f}ms")
        else:
            print("Path planning failed!")
            self.path_planned = False

    def _should_replan(self, fused_map, dt: float) -> Tuple[bool, str]:
        """Determine if replanning is needed."""
        current_time = time.time() - self.start_time

        # Periodic replanning
        self.replan_counter += 1
        if self.replan_counter >= self.replan_interval:
            self.replan_counter = 0
            return True, "periodic"

        # Tracking error trigger
        if self.path_interpolated:
            # Find closest point on path
            min_dist = float('inf')
            for wp in self.path_interpolated:
                dist = euclidean_distance((self.state.x, self.state.y), wp)
                if dist < min_dist:
                    min_dist = dist

            if min_dist > self.max_tracking_error:
                return True, f"tracking_error_{min_dist:.2f}m"

        # Dynamic obstacle detected
        if self.dynamic_obstacles:
            # Check if any dynamic obstacle is near planned path
            for obs_x, obs_y, obs_rad in self.dynamic_obstacles:
                for wp in self.path_interpolated[self.current_waypoint_idx:self.current_waypoint_idx+10]:
                    if euclidean_distance((obs_x, obs_y), wp) < obs_rad + 0.3:
                        return True, f"dynamic_obstacle_near_path"

        # Collision imminent
        ds_values = []
        for ds in self.distance_sensors:
            ds_values.append(ds.getValue())
        if self._check_collision(ds_values):
            return True, "collision_imminent"

        return False, ""

    def _update_occupancy_map(self, fused_map):
        """Update occupancy map from fused perception."""
        if self.occupancy_map is not None:
            self.occupancy_map.grid = fused_map.occupancy_grid
            self.occupancy_map.obstacle_map = (fused_map.occupancy_grid >= 0.5).astype(np.uint8)

        # Add perceived obstacles to map
        self._add_obstacles_to_map(fused_map)

        # Update dynamic obstacle tracking
        self._update_dynamic_obstacles(fused_map)

    def _update_occupancy_map(self, fused_map):
        """Update occupancy map from fused perception."""
        if self.occupancy_map is not None:
            self.occupancy_map.grid = fused_map.occupancy_grid
            self.occupancy_map.obstacle_map = (fused_map.occupancy_grid >= 0.5).astype(np.uint8)

    def _control_step(self, dt: float) -> Tuple[float, float]:
        """Execute control step and return wheel velocities."""
        if not self.path_planned or self.goal is None:
            return 0.0, 0.0

        current_time = time.time() - self.start_time

        # Execute path tracking
        v, omega, complete = self.controller.step(self.state, dt, current_time)

        if complete:
            print("Goal reached!")
            self._log_results()
            self._export_results()
            return 0.0, 0.0

        # Convert to wheel velocities
        left_vel, right_vel = self.kinematics.inverse_kinematics(v, omega)

        # Apply motor limits
        max_vel = self.robot_config.max_linear_velocity / self.robot_config.wheel_radius + \
                  self.robot_config.max_angular_velocity * self.robot_config.wheel_base / \
                  (2 * self.robot_config.wheel_radius)

        left_vel = np.clip(left_vel, -max_vel, max_vel)
        right_vel = np.clip(right_vel, -max_vel, max_vel)

        # Log control data
        self.metrics.control_effort += abs(v) + abs(omega)
        self.control_log.append({
            'time': current_time,
            'v_cmd': v,
            'omega_cmd': omega,
            'left_vel': left_vel,
            'right_vel': right_vel,
            'state_x': self.state.x,
            'state_y': self.state.y,
            'state_theta': self.state.theta
        })

        return left_vel, right_vel

    def _log_results(self):
        """Log final results for analysis."""
        if self.planned_path_log and self.trajectory_log:
            from python_analysis.utils.helpers import compute_tracking_error, compute_path_length

            planned_len = compute_path_length(self.planned_path_log)
            actual_len = compute_path_length([(x, y) for x, y, t in self.trajectory_log])

            error_metrics = compute_tracking_error(
                self.planned_path_log, [(x, y) for x, y, t in self.trajectory_log])

            self.metrics.planned_path_length = planned_len
            self.metrics.actual_path_length = actual_len
            self.metrics.path_efficiency = planned_len / actual_len * 100 if actual_len > 0 else 0
            self.metrics.time_to_goal = time.time() - self.start_time
            self.metrics.max_tracking_error = error_metrics['max_error']
            self.metrics.rmse_tracking_error = error_metrics['rmse']
            self.metrics.mean_tracking_error = error_metrics['mean_error']
            self.metrics.std_tracking_error = error_metrics['std_error']
            self.metrics.goal_reached = True

            print("\n=== NAVIGATION RESULTS ===")
            print(f"Planned path length: {planned_len:.2f} m")
            print(f"Actual path length: {actual_len:.2f} m")
            print(f"Path efficiency: {self.metrics.path_efficiency:.1f}%")
            print(f"Max tracking error: {error_metrics['max_error']:.3f} m")
            print(f"RMSE tracking error: {error_metrics['rmse']:.3f} m")
            print(f"Mean tracking error: {error_metrics['mean_error']:.3f} m")
            print(f"Std tracking error: {error_metrics['std_error']:.3f} m")
            print(f"Time to goal: {self.metrics.time_to_goal:.1f} s")
            print(f"Number of replans: {self.metrics.num_replans}")
            print(f"Replan reasons: {', '.join(self.metrics.replan_reasons)}")
            print(f"Control effort: {self.metrics.control_effort:.2f}")

    def _export_results(self):
        """Export results for analysis."""
        # Export trajectory
        trajectory_file = os.path.join(self.export_dir, f"trajectory_{int(self.start_time)}.json")
        trajectory_data = {
            'planned_path': self.planned_path_log,
            'actual_trajectory': self.trajectory_log,
            'control_log': self.control_log,
            'replan_log': self.replan_log,
            'metrics': asdict(self.metrics),
            'config': {
                'grid_resolution': self.env_config.grid_resolution,
                'world_size': self.env_config.world_size,
                'robot_config': asdict(self.robot_config),
                'pid_config': asdict(self.pid_config)
            }
        }
        with open(trajectory_file, 'w') as f:
            json.dump(trajectory_data, f, indent=2)
        print(f"Results exported to: {trajectory_file}")

        # Export performance table as CSV
        import csv
        csv_file = os.path.join(self.export_dir, f"performance_{int(self.start_time)}.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Planned Path Length (m)', f"{self.metrics.planned_path_length:.2f}"])
            writer.writerow(['Actual Path Length (m)', f"{self.metrics.actual_path_length:.2f}"])
            writer.writerow(['Path Efficiency (%)', f"{self.metrics.path_efficiency:.1f}"])
            writer.writerow(['Time to Goal (s)', f"{self.metrics.time_to_goal:.1f}"])
            writer.writerow(['Max Tracking Error (m)', f"{self.metrics.max_tracking_error:.3f}"])
            writer.writerow(['RMSE Tracking Error (m)', f"{self.metrics.rmse_tracking_error:.3f}"])
            writer.writerow(['Mean Tracking Error (m)', f"{self.metrics.mean_tracking_error:.3f}"])
            writer.writerow(['Std Tracking Error (m)', f"{self.metrics.std_tracking_error:.3f}"])
            writer.writerow(['Number of Replans', self.metrics.num_replans])
            writer.writerow(['Goal Reached', self.metrics.goal_reached])
            writer.writerow(['Collisions', self.metrics.collisions])
            writer.writerow(['Control Effort', f"{self.metrics.control_effort:.2f}"])
        print(f"Performance table exported to: {csv_file}")

    def run(self):
        """Main control loop."""
        print("Starting Component A controller...")

        # Wait for first sensor readings
        for _ in range(10):
            if self.robot.step(self.timestep) == -1:
                return
            self._read_sensors()

        # Initialize planning
        self._initialize_planning()

        print("Entering main loop...")

        while self.robot.step(self.timestep) != -1:
            current_time = time.time()
            dt = current_time - self.last_time
            self.last_time = current_time

            # Read sensors
            lidar_ranges, camera_image, pose, ds_values = self._read_sensors()

            # Update state
            self._update_state(pose)
            # Log trajectory with timestamp
            self.trajectory_log.append((self.state.x, self.state.y, time.time() - self.start_time))

            # Check collision
            if self._check_collision(ds_values):
                self.metrics.collisions += 1
                print("Collision detected!")
                self.emergency_stop = True

            if self.emergency_stop:
                if self.left_motor and self.right_motor:
                    self.left_motor.setVelocity(0.0)
                    self.right_motor.setVelocity(0.0)
                continue

            # Perception pipeline
            fused_map = self.perception.process(
                lidar_ranges, camera_image, pose, current_time
            )

            # Update occupancy map
            self._update_occupancy_map(fused_map)

            # Check if replanning is needed
            should_replan, reason = self._should_replan(fused_map, dt)
            if should_replan:
                self._replan(reason)

            # Control
            left_vel, right_vel = self._control_step(dt)

            # Apply motor commands
            if self.left_motor and self.right_motor:
                self.left_motor.setVelocity(left_vel)
                self.right_motor.setVelocity(right_vel)

            # Debug output
            if len(self.trajectory_log) % 100 == 0:
                dist_to_goal = euclidean_distance((self.state.x, self.state.y), self.goal)
                print(f"Pos: ({self.state.x:.2f}, {self.state.y:.2f}), "
                      f"Goal dist: {dist_to_goal:.2f}, v={left_vel:.2f}/{right_vel:.2f}, "
                      f"replans={self.metrics.num_replans}")


def main():
    """Entry point for Webots controller."""
    controller = RobotControllerA()
    controller.run()


if __name__ == "__main__":
    main()