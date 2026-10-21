"""
Main Robot Controller for ASR700 Component A.

Integrates perception, planning, and control for autonomous navigation
in Webots simulation.
"""

import sys
import os
import time
import numpy as np
from typing import List, Tuple, Optional

# Add python_analysis to path for imports
controller_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(controller_dir)))
sys.path.insert(0, project_root)

from python_analysis.utils.config import (SensorConfig, EnvironmentConfig,
                                           RobotConfig, PIDConfig, PathPlanningConfig)
from python_analysis.utils.helpers import (normalize_angle, angle_difference,
                                           euclidean_distance, interpolate_path)
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
        self.trajectory_log: List[Tuple[float, float]] = []
        self.planned_path_log: List[Tuple[float, float]] = []

        # Planning state
        self.path_planned = False
        self.replan_counter = 0
        self.replan_interval = 50  # Replan every N steps

    def _init_devices(self):
        """Initialize Webots devices."""
        # LiDAR
        self.lidar = self.robot.getDevice('lidar')
        if self.lidar:
            self.lidar.enable(self.timestep)
            self.lidar.enablePointCloud()

        # Camera
        self.camera = self.robot.getDevice('camera')
        if self.camera:
            self.camera.enable(self.timestep)
            self.camera.recognitionEnable(self.timestep)

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

        print("Devices initialized")

    def _read_sensors(self) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
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

        return lidar_ranges, camera_image, pose

    def _update_state(self, pose: Tuple[float, float, float]):
        """Update internal state estimate."""
        self.state.x = pose[0]
        self.state.y = pose[1]
        self.state.theta = normalize_angle(pose[2])

        # Log trajectory
        self.trajectory_log.append((self.state.x, self.state.y))

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

        # Plan initial path
        self._replan()

    def _replan(self):
        """Replan path using current occupancy map."""
        if self.occupancy_map is None or self.goal is None:
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
            self.planned_path = path
            # Interpolate for smoother tracking
            self.path_interpolated = interpolate_path(path, step_size=0.1)
            self.controller.set_path(self.path_interpolated)
            self.current_waypoint_idx = 0
            self.path_planned = True

            # Log for comparison
            self.planned_path_log = path.copy()

            print(f"Path planned: {len(path)} waypoints, length={stats.path_length:.2f}m, "
                  f"nodes={stats.nodes_explored}, time={stats.computation_time_ms:.1f}ms")
        else:
            print("Path planning failed!")
            self.path_planned = False

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
            return 0.0, 0.0

        # Convert to wheel velocities
        left_vel, right_vel = self.kinematics.inverse_kinematics(v, omega)

        # Apply motor limits
        max_vel = self.robot_config.max_linear_velocity / self.robot_config.wheel_radius + \
                  self.robot_config.max_angular_velocity * self.robot_config.wheel_base / \
                  (2 * self.robot_config.wheel_radius)

        left_vel = np.clip(left_vel, -max_vel, max_vel)
        right_vel = np.clip(right_vel, -max_vel, max_vel)

        return left_vel, right_vel

    def _log_results(self):
        """Log final results for analysis."""
        if self.planned_path_log and self.trajectory_log:
            from python_analysis.utils.helpers import compute_tracking_error, compute_path_length

            planned_len = compute_path_length(self.planned_path_log)
            actual_len = compute_path_length(self.trajectory_log)

            error_metrics = compute_tracking_error(
                self.planned_path_log, self.trajectory_log)

            print("\n=== NAVIGATION RESULTS ===")
            print(f"Planned path length: {planned_len:.2f} m")
            print(f"Actual path length: {actual_len:.2f} m")
            print(f"Path efficiency: {planned_len/actual_len*100:.1f}%")
            print(f"Max tracking error: {error_metrics['max_error']:.3f} m")
            print(f"RMSE tracking error: {error_metrics['rmse']:.3f} m")
            print(f"Mean tracking error: {error_metrics['mean_error']:.3f} m")
            print(f"Time to goal: {time.time() - self.start_time:.1f} s")

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
            lidar_ranges, camera_image, pose = self._read_sensors()

            # Update state
            self._update_state(pose)

            # Perception pipeline
            fused_map = self.perception.process(
                lidar_ranges, camera_image, pose, current_time
            )

            # Update occupancy map
            self._update_occupancy_map(fused_map)

            # Periodic replanning
            self.replan_counter += 1
            if self.replan_counter >= self.replan_interval:
                self._replan()
                self.replan_counter = 0

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
                      f"Goal dist: {dist_to_goal:.2f}, v={left_vel:.2f}/{right_vel:.2f}")


def main():
    """Entry point for Webots controller."""
    controller = RobotControllerA()
    controller.run()


if __name__ == "__main__":
    main()