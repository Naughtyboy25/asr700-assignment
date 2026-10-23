"""
Leader Controller for ASR700 Component B.

Implements the leader agent behavior for multi-agent swarm system.
Coordinates with followers via consensus and PSO-based task allocation.
"""

import sys
import os
import time
import numpy as np
from typing import List, Tuple, Optional, Dict

# Add project root to path
controller_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(controller_dir)))
sys.path.insert(0, project_root)

from python_analysis.utils.config import (SensorConfig, EnvironmentConfig,
                                           RobotConfig, SwarmConfig)
from python_analysis.utils.helpers import (normalize_angle, angle_difference,
                                           euclidean_distance, PIDController)
from webots_project.controllers.swarm_agents.consensus import (ConsensusProtocol,
                                                                AgentState,
                                                                LeaderElection,
                                                                ConsensusType)
from webots_project.controllers.swarm_agents.swarm_intelligence import (ParticleSwarmOptimization,
                                                                         CoverageACO)
from webots_project.controllers.swarm_agents.ml_policy import (NavigationCNN,
                                                                DeployablePolicy)

# Webots imports
try:
    from controller import Robot, Lidar, Camera, Motor, GPS, InertialUnit, Emitter, Receiver
except ImportError:
    class Robot:
        def __init__(self): self.timestep = 32
        def step(self, ts): return -1
        def getBasicTimeStep(self): return 32
    class Lidar:
        def __init__(self, n): pass
        def enable(self, ts): pass
        def getRangeImage(self): return []
    class Camera:
        def __init__(self, n): pass
        def enable(self, ts): pass
        def getImageArray(self): return None
    class Motor:
        def __init__(self, n): pass
        def setPosition(self, p): pass
        def setVelocity(self, v): pass
    class GPS:
        def __init__(self, n): pass
        def enable(self, ts): pass
        def getValues(self): return [0,0,0]
    class InertialUnit:
        def __init__(self, n): pass
        def enable(self, ts): pass
        def getRollPitchYaw(self): return [0,0,0]
    class Emitter:
        def __init__(self, n): pass
        def send(self, data): pass
    class Receiver:
        def __init__(self, n): pass
        def enable(self, ts): pass
        def getQueueLength(self): return 0
        def getData(self): return b''
        def nextPacket(self): pass


class LeaderController:
    """
    Leader agent controller for swarm coordination.

    Responsibilities:
    - Global task planning (PSO/ACO for coverage)
    - Consensus coordination
    - Communication with followers
    - Goal assignment
    """

    def __init__(self):
        # Don't create Robot here - defer to main()
        self.robot = None
        self.timestep = 32

        # Configuration
        self.sensor_config = SensorConfig()
        self.env_config = EnvironmentConfig()
        self.robot_config = RobotConfig()
        self.swarm_config = SwarmConfig()

        # State
        self.agent_id = 0  # Leader is agent 0
        self.position = np.array([0.0, 0.0])
        self.heading = 0.0
        self.velocity = np.array([0.0, 0.0])

        # Swarm state (known followers)
        self.follower_states: Dict[int, AgentState] = {}
        self.num_followers = self.swarm_config.num_agents - 1

        # Consensus
        self.consensus = ConsensusProtocol(self.swarm_config)
        self.consensus_value = np.array([0.0, 0.0])  # Target position

        # Task allocation
        self.pso = ParticleSwarmOptimization(self.swarm_config)
        self.coverage_targets: List[Tuple[float, float]] = []

        # Control
        self.heading_pid = PIDController(2.0, 0.1, 0.5, (-2.0, 2.0))
        self.distance_pid = PIDController(1.0, 0.05, 0.2, (-0.12, 0.12))

        # ML policy (optional)
        self.ml_policy = None
        self.use_ml = False

        # Logging
        self.trajectory = []
        self.consensus_history = []
        self.start_time = time.time()
        self.last_time = self.start_time

    def _init(self):
        """Initialize Robot and devices (called from main)."""
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        # Devices
        self._init_devices()

    def _init_devices(self):
        """Initialize Webots devices."""
        # Sensors
        self.lidar = self.robot.getDevice('lidar')
        if self.lidar:
            self.lidar.enable(self.timestep)

        self.camera = self.robot.getDevice('camera')
        if self.camera:
            self.camera.enable(self.timestep)

        # Motors
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        if self.left_motor and self.right_motor:
            self.left_motor.setPosition(float('inf'))
            self.right_motor.setPosition(float('inf'))

        # Localization
        self.gps = self.robot.getDevice('gps')
        if self.gps:
            self.gps.enable(self.timestep)

        self.imu = self.robot.getDevice('inertial unit')
        if self.imu:
            self.imu.enable(self.timestep)

        # Communication
        self.emitter = self.robot.getDevice('emitter')
        if self.emitter:
            self.emitter.setChannel(1)
            self.emitter.setRange(self.swarm_config.communication_range)

        self.receiver = self.robot.getDevice('receiver')
        if self.receiver:
            self.receiver.enable(self.timestep)
            self.receiver.setChannel(1)

        print("Leader devices initialized")

    def _read_sensors(self) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
        """Read sensor data."""
        # LiDAR
        lidar_ranges = np.array([])
        if self.lidar:
            raw = self.lidar.getRangeImage()
            if raw:
                lidar_ranges = np.array(raw[::-1])  # Webots returns reversed

        # Camera
        camera_image = np.zeros((self.sensor_config.camera_height,
                                 self.sensor_config.camera_width, 3), dtype=np.uint8)
        if self.camera:
            img = self.camera.getImageArray()
            if img:
                camera_image = np.array(img, dtype=np.uint8)

        # Pose
        pose = (0.0, 0.0, 0.0)
        if self.gps and self.imu:
            gps_vals = self.gps.getValues()
            imu_vals = self.imu.getRollPitchYaw()
            pose = (gps_vals[0], gps_vals[1], imu_vals[2])

        return lidar_ranges, camera_image, pose

    def _update_self_state(self, pose: Tuple[float, float, float]):
        """Update internal state estimate."""
        self.position = np.array([pose[0], pose[1]])
        self.heading = normalize_angle(pose[2])
        self.trajectory.append(self.position.copy())

    def _broadcast_state(self):
        """Broadcast state to followers."""
        if not self.emitter:
            return

        # Simple message: agent_id, x, y, theta, consensus_value_x, consensus_value_y
        msg = [self.agent_id,
               self.position[0], self.position[1], self.heading,
               self.consensus_value[0], self.consensus_value[1]]
        self.emitter.send(",".join(f"{v:.6f}" for v in msg))

    def _receive_messages(self):
        """Receive messages from followers."""
        if not self.receiver:
            return

        while self.receiver.getQueueLength() > 0:
            data = self.receiver.getString()
            self.receiver.nextPacket()

            try:
                parts = data.split(",")
                if len(parts) >= 6:
                    msg = [float(p) for p in parts[:6]]
                    follower_id = int(msg[0])
                    pos = (msg[1], msg[2])
                    heading = msg[3]
                    value = np.array([msg[4], msg[5]])
            except (ValueError, IndexError):
                return

                # Update follower state
                self.follower_states[follower_id] = AgentState(
                    agent_id=follower_id,
                    position=pos,
                    heading=heading,
                    vector_value=value
                )

    def _run_consensus(self):
        """Run consensus with followers."""
        # Collect all agent states
        agents = [AgentState(self.agent_id, tuple(self.position),
                             heading=self.heading,
                             vector_value=self.consensus_value)]

        for fid, fstate in self.follower_states.items():
            agents.append(fstate)

        # Run average consensus on target position
        final_agents, info = self.consensus.run_consensus(
            agents, ConsensusType.AVERAGE, max_iterations=10, dt=0.1
        )

        # Update consensus value
        self.consensus_value = info['final_consensus']
        self.consensus_history.append(self.consensus_value.copy())

        # Update follower target assignments
        for i, agent in enumerate(final_agents):
            if agent.agent_id != self.agent_id:
                if agent.agent_id in self.follower_states:
                    self.follower_states[agent.agent_id].vector_value = agent.vector_value

    def _plan_coverage(self):
        """Plan coverage targets using PSO."""
        # Create occupancy grid from environment (simplified)
        grid_w = int(self.env_config.world_size[0] / self.env_config.grid_resolution)
        grid_h = int(self.env_config.world_size[1] / self.env_config.grid_resolution)
        grid = np.zeros((grid_h, grid_w))

        # Add known obstacles (would come from perception in real implementation)
        # For now, assume empty grid

        # Define fitness for coverage
        agent_starts = [tuple(self.position)]
        for fstate in self.follower_states.values():
            agent_starts.append(fstate.position)

        def coverage_fitness(waypoints_flat):
            waypoints = waypoints_flat.reshape(-1, 2)
            all_waypoints = np.vstack([agent_starts, waypoints])

            # Coverage metric
            covered = 0
            sensor_range = 1.0
            for r in range(grid_h):
                for c in range(grid_w):
                    if grid[r, c] == 0:
                        wx, wy = c * self.env_config.grid_resolution, r * self.env_config.grid_resolution
                        for wp in all_waypoints:
                            if euclidean_distance((wx, wy), wp) <= sensor_range:
                                covered += 1
                                break

            return -covered  # Minimize negative coverage

        # Optimize waypoints for followers
        num_waypoints = self.num_followers * 2  # 2 waypoints per follower
        # Bounds for each waypoint (x, y), replicated for all waypoints
        lower = np.tile([0.5, 0.5], num_waypoints)
        upper = np.tile([9.5, 9.5], num_waypoints)
        bounds = (lower, upper)

        self.pso.bounds = bounds
        result = self.pso.optimize(coverage_fitness, dim=num_waypoints * 2, verbose=False)

        # Extract waypoints
        waypoints = result.best_position.reshape(-1, 2)
        self.coverage_targets = [tuple(wp) for wp in waypoints]

        # Use a real coverage waypoint as the leader's target
        if self.coverage_targets:
            self.consensus_value = np.asarray(self.coverage_targets[0], dtype=float)

        print(f"Leader: Planned {len(self.coverage_targets)} coverage targets")

    def _assign_targets(self):
        """Assign coverage targets to followers."""
        # Simple assignment: closest target to each follower
        for i, (fid, fstate) in enumerate(self.follower_states.items()):
            if i < len(self.coverage_targets):
                target = self.coverage_targets[i]
                fstate.vector_value = np.array(target)
                # Update in our records
                self.follower_states[fid] = fstate

    def _control_step(self, dt: float) -> Tuple[float, float]:
        """Compute motor velocities for leader."""
        # Leader moves to its consensus target
        target = self.consensus_value

        dx = target[0] - self.position[0]
        dy = target[1] - self.position[1]
        distance = np.hypot(dx, dy)

        if distance < 0.2:
            return 0.0, 0.0

        desired_heading = np.arctan2(dy, dx)
        heading_error = angle_difference(desired_heading, self.heading)

        # PID control
        v = self.distance_pid.step(distance, dt)
        omega = self.heading_pid.step(heading_error, dt)

        return v, omega

    def run(self):
        """Main control loop."""
        print("Leader controller starting...")

        # Wait for sensors
        for _ in range(10):
            if self.robot.step(self.timestep) == -1:
                return
            self._read_sensors()

        # Initial coverage planning
        self._plan_coverage()
        self._assign_targets()

        while self.robot.step(self.timestep) != -1:
            current_time = time.time()
            dt = current_time - self.last_time
            self.last_time = current_time

            # Read sensors
            lidar_ranges, camera_image, pose = self._read_sensors()

            # Update state
            self._update_self_state(pose)

            # Communication
            self._receive_messages()
            self._broadcast_state()

            # Consensus
            if len(self.follower_states) > 0:
                self._run_consensus()

            # Periodic replanning
            if len(self.trajectory) % 200 == 0:
                self._plan_coverage()
                self._assign_targets()

            # Control
            v, omega = self._control_step(dt)

            # Convert to wheel velocities
            R = self.robot_config.wheel_radius
            L = self.robot_config.wheel_base
            left = (2 * v - omega * L) / (2 * R)
            right = (2 * v + omega * L) / (2 * R)

            # Apply
            if self.left_motor and self.right_motor:
                max_vel = 6.28  # rad/s
                self.left_motor.setVelocity(np.clip(left, -max_vel, max_vel))
                self.right_motor.setVelocity(np.clip(right, -max_vel, max_vel))

            # Debug
            if len(self.trajectory) % 100 == 0:
                print(f"Leader: pos=({self.position[0]:.2f}, {self.position[1]:.2f}), "
                      f"target=({self.consensus_value[0]:.2f}, {self.consensus_value[1]:.2f}), "
                      f"followers={len(self.follower_states)}")


def main():
    controller = LeaderController()
    controller._init()
    controller.run()


if __name__ == "__main__":
    main()