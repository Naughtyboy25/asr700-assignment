"""
Follower Controller for ASR700 Component B.

Implements follower agent behavior for multi-agent swarm system.
Participates in consensus and executes assigned tasks.
"""

import sys
import os
import time
import numpy as np
from typing import Dict, Tuple, Optional

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
                                                                ConsensusType)

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


class FollowerController:
    """
    Follower agent controller for swarm coordination.

    Responsibilities:
    - Participate in consensus
    - Execute assigned tasks
    - Local obstacle avoidance
    - Report state to leader
    """

    def __init__(self, agent_id: int = 1):
        # Don't create Robot here - defer to main()
        self.robot = None
        self.timestep = 32

        # Configuration
        self.sensor_config = SensorConfig()
        self.env_config = EnvironmentConfig()
        self.robot_config = RobotConfig()
        self.swarm_config = SwarmConfig()

        # Agent identity
        self.agent_id = agent_id

        # State
        self.position = np.array([0.0, 0.0])
        self.heading = 0.0
        self.target: Optional[np.ndarray] = None

        # Consensus
        self.consensus = ConsensusProtocol(self.swarm_config)
        self.leader_state: Optional[AgentState] = None
        self.follower_states: Dict[int, AgentState] = {}

        # Control
        self.heading_pid = PIDController(3.0, 0.1, 0.5, (-2.0, 2.0))
        self.distance_pid = PIDController(1.5, 0.05, 0.3, (-0.12, 0.12))

        # Local obstacle avoidance
        self.avoidance_active = False
        self.avoidance_heading = 0.0

        # Logging
        self.trajectory = []
        self.consensus_history = []
        self.start_time = time.time()
        self.last_time = self.start_time

    def _init(self):
        """Initialize Robot and devices (called from main)."""
        if self.robot is None:
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

        print(f"Follower {self.agent_id} devices initialized")

    def _read_sensors(self) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
        """Read sensor data."""
        # LiDAR
        lidar_ranges = np.array([])
        if self.lidar:
            raw = self.lidar.getRangeImage()
            if raw:
                lidar_ranges = np.array(raw[::-1])

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
        """Update internal state."""
        self.position = np.array([pose[0], pose[1]])
        self.heading = normalize_angle(pose[2])
        self.trajectory.append(self.position.copy())

    def _check_obstacle_avoidance(self, lidar_ranges: np.ndarray) -> Tuple[bool, float]:
        """
        Simple obstacle avoidance using LiDAR.

        Returns:
            (avoidance_active, avoidance_heading)
        """
        if len(lidar_ranges) == 0:
            return False, 0.0

        # Check front sector (±45 degrees)
        n = len(lidar_ranges)
        front_start = n // 2 - n // 8
        front_end = n // 2 + n // 8
        front_ranges = lidar_ranges[front_start:front_end]

        min_front = np.min(front_ranges) if len(front_ranges) > 0 else 5.0

        if min_front < 0.4:  # Obstacle too close
            # Find best direction to turn
            left_ranges = lidar_ranges[:n//2]
            right_ranges = lidar_ranges[n//2:]

            left_min = np.min(left_ranges) if len(left_ranges) > 0 else 0
            right_min = np.min(right_ranges) if len(right_ranges) > 0 else 0

            # Turn toward more open side
            if left_min > right_min:
                avoidance_heading = self.heading + np.pi/2  # Turn left
            else:
                avoidance_heading = self.heading - np.pi/2  # Turn right

            return True, normalize_angle(avoidance_heading)

        return False, 0.0

    def _broadcast_state(self):
        """Broadcast state to leader and other followers."""
        if not self.emitter:
            return

        # Current target (from consensus or assigned)
        target_val = self.target if self.target is not None else self.position

        msg = [self.agent_id,
               self.position[0], self.position[1], self.heading,
               target_val[0], target_val[1]]
        self.emitter.send(",".join(f"{v:.6f}" for v in msg))

    def _receive_messages(self):
        """Receive messages from leader and other followers."""
        if not self.receiver:
            return

        while self.receiver.getQueueLength() > 0:
            data = self.receiver.getString()
            self.receiver.nextPacket()

            try:
                parts = data.split(",")
                if len(parts) >= 6:
                    msg = [float(p) for p in parts[:6]]
                    sender_id = int(msg[0])
                    pos = (msg[1], msg[2])
                    heading = msg[3]
                    value = np.array([msg[4], msg[5]])
            except (ValueError, IndexError):
                return

                if sender_id == 0:  # Leader
                    self.leader_state = AgentState(
                        agent_id=0, position=pos, heading=heading, vector_value=value
                    )
                    # Update target from leader
                    if value is not None:
                        self.target = value
                else:  # Other follower
                    self.follower_states[sender_id] = AgentState(
                        agent_id=sender_id, position=pos, heading=heading, vector_value=value
                    )

    def _run_consensus(self):
        """Participate in consensus with neighbors."""
        # Collect all known agents
        agents = [AgentState(self.agent_id, tuple(self.position),
                             heading=self.heading,
                             vector_value=self.target if self.target is not None else self.position)]

        if self.leader_state:
            agents.append(self.leader_state)

        for fstate in self.follower_states.values():
            agents.append(fstate)

        # Run consensus
        final_agents, info = self.consensus.run_consensus(
            agents, ConsensusType.AVERAGE, max_iterations=10, dt=0.1
        )

        # Update our target from consensus
        for agent in final_agents:
            if agent.agent_id == self.agent_id:
                self.target = agent.vector_value
                self.consensus_history.append(self.target.copy())
                break

    def _control_step(self, dt: float, lidar_ranges: np.ndarray) -> Tuple[float, float]:
        """Compute motor velocities."""
        # Check obstacle avoidance
        self.avoidance_active, self.avoidance_heading = self._check_obstacle_avoidance(lidar_ranges)

        if self.target is None:
            return 0.0, 0.0

        # Determine target heading
        if self.avoidance_active:
            target_heading = self.avoidance_heading
            distance = 1.0  # Arbitrary distance for avoidance
        else:
            dx = self.target[0] - self.position[0]
            dy = self.target[1] - self.position[1]
            distance = np.hypot(dx, dy)
            target_heading = np.arctan2(dy, dx)

        if distance < 0.15 and not self.avoidance_active:
            return 0.0, 0.0

        heading_error = angle_difference(target_heading, self.heading)

        # PID control
        v = self.distance_pid.step(distance, dt)
        omega = self.heading_pid.step(heading_error, dt)

        # Reduce speed during avoidance
        if self.avoidance_active:
            v *= 0.5

        return v, omega

    def run(self):
        """Main control loop."""
        print(f"Follower {self.agent_id} controller starting...")

        # Wait for sensors
        for _ in range(10):
            if self.robot.step(self.timestep) == -1:
                return
            self._read_sensors()

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
            if self.leader_state is not None:
                self._run_consensus()

            # Control
            v, omega = self._control_step(dt, lidar_ranges)

            # Convert to wheel velocities
            R = self.robot_config.wheel_radius
            L = self.robot_config.wheel_base
            left = (2 * v - omega * L) / (2 * R)
            right = (2 * v + omega * L) / (2 * R)

            # Apply
            if self.left_motor and self.right_motor:
                max_vel = 6.28
                self.left_motor.setVelocity(np.clip(left, -max_vel, max_vel))
                self.right_motor.setVelocity(np.clip(right, -max_vel, max_vel))

            # Debug
            if len(self.trajectory) % 100 == 0:
                target_str = f"({self.target[0]:.2f}, {self.target[1]:.2f})" if self.target is not None else "None"
                print(f"Follower {self.agent_id}: pos=({self.position[0]:.2f}, {self.position[1]:.2f}), "
                      f"target={target_str}, avoid={self.avoidance_active}")


def main():
    # Get agent ID from robot name or environment
    # In Webots, each robot instance has a unique name
    # Create a single Robot instance (Webots allows only one per controller)
    robot = Robot()
    robot_name = robot.getName()

    # Parse agent ID from name (e.g., "e-puck_follower_1" -> 1)
    agent_id = 1
    if 'follower' in robot_name:
        try:
            agent_id = int(robot_name.split('_')[-1])
        except:
            agent_id = 1

    controller = FollowerController(agent_id)
    controller.robot = robot
    controller._init()
    controller.run()


if __name__ == "__main__":
    main()