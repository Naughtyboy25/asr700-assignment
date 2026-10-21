"""
Differential Drive Robot Kinematics for ASR700 Component A.

Implements forward and inverse kinematics for a differential drive robot
(e-puck compatible).
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple
from python_analysis.utils.config import RobotConfig


@dataclass(frozen=True)
class RobotState:
    """Robot state representation."""
    x: float = 0.0          # World x position (m)
    y: float = 0.0          # World y position (m)
    theta: float = 0.0      # Heading angle (rad), 0 = +x axis
    v: float = 0.0          # Linear velocity (m/s)
    omega: float = 0.0      # Angular velocity (rad/s)
    left_wheel_vel: float = 0.0   # Left wheel velocity (rad/s)
    right_wheel_vel: float = 0.0  # Right wheel velocity (rad/s)


class DifferentialDriveKinematics:
    """
    Differential drive kinematics model.

    Forward kinematics: wheel velocities -> robot velocities -> pose
    Inverse kinematics: robot velocities -> wheel velocities
    """

    def __init__(self, config: RobotConfig = None):
        self.config = config or RobotConfig()
        self.R = self.config.wheel_radius
        self.L = self.config.wheel_base

    def forward_kinematics(self, left_wheel_vel: float, right_wheel_vel: float) -> Tuple[float, float]:
        """
        Compute robot linear and angular velocity from wheel velocities.

        Args:
            left_wheel_vel: Left wheel angular velocity (rad/s)
            right_wheel_vel: Right wheel angular velocity (rad/s)

        Returns:
            (linear_velocity, angular_velocity) in (m/s, rad/s)
        """
        v = self.R * (left_wheel_vel + right_wheel_vel) / 2.0
        omega = self.R * (right_wheel_vel - left_wheel_vel) / self.L
        return v, omega

    def inverse_kinematics(self, v: float, omega: float) -> Tuple[float, float]:
        """
        Compute wheel velocities from desired robot velocities.

        Args:
            v: Desired linear velocity (m/s)
            omega: Desired angular velocity (rad/s)

        Returns:
            (left_wheel_vel, right_wheel_vel) in rad/s
        """
        left = (2 * v - omega * self.L) / (2 * self.R)
        right = (2 * v + omega * self.L) / (2 * self.R)
        return left, right

    def integrate_odometry(self, state: RobotState, dt: float) -> RobotState:
        """
        Integrate robot pose using odometry (forward kinematics + integration).

        Args:
            state: Current robot state with wheel velocities
            dt: Time step (s)

        Returns:
            Updated robot state
        """
        v, omega = self.forward_kinematics(state.left_wheel_vel, state.right_wheel_vel)

        # Exact integration for constant velocities
        if abs(omega) < 1e-6:
            # Straight line motion
            new_x = state.x + v * dt * np.cos(state.theta)
            new_y = state.y + v * dt * np.sin(state.theta)
            new_theta = state.theta
        else:
            # Circular arc motion
            radius = v / omega
            new_x = state.x + radius * (np.sin(state.theta + omega * dt) - np.sin(state.theta))
            new_y = state.y - radius * (np.cos(state.theta + omega * dt) - np.cos(state.theta))
            new_theta = state.theta + omega * dt

        # Normalize angle
        new_theta = np.arctan2(np.sin(new_theta), np.cos(new_theta))

        return RobotState(
            x=new_x,
            y=new_y,
            theta=new_theta,
            v=v,
            omega=omega,
            left_wheel_vel=state.left_wheel_vel,
            right_wheel_vel=state.right_wheel_vel
        )

    def integrate_velocity(self, state: RobotState, v: float, omega: float, dt: float) -> RobotState:
        """
        Integrate robot pose using commanded velocities.

        Args:
            state: Current robot state
            v: Commanded linear velocity (m/s)
            omega: Commanded angular velocity (rad/s)
            dt: Time step (s)

        Returns:
            Updated robot state
        """
        if abs(omega) < 1e-6:
            new_x = state.x + v * dt * np.cos(state.theta)
            new_y = state.y + v * dt * np.sin(state.theta)
            new_theta = state.theta
        else:
            radius = v / omega
            new_x = state.x + radius * (np.sin(state.theta + omega * dt) - np.sin(state.theta))
            new_y = state.y - radius * (np.cos(state.theta + omega * dt) - np.cos(state.theta))
            new_theta = state.theta + omega * dt

        new_theta = np.arctan2(np.sin(new_theta), np.cos(new_theta))
        left, right = self.inverse_kinematics(v, omega)

        return RobotState(
            x=new_x,
            y=new_y,
            theta=new_theta,
            v=v,
            omega=omega,
            left_wheel_vel=left,
            right_wheel_vel=right
        )

    def compute_wheel_velocities(self, v: float, omega: float) -> Tuple[float, float]:
        """
        Compute wheel velocities with saturation limits.

        Args:
            v: Desired linear velocity (m/s)
            omega: Desired angular velocity (rad/s)

        Returns:
            (left_wheel_vel, right_wheel_vel) clamped to motor limits
        """
        left, right = self.inverse_kinematics(v, omega)

        # Convert to wheel angular velocity limits
        max_wheel_vel = self.config.max_linear_velocity / self.R + \
                        self.config.max_angular_velocity * self.L / (2 * self.R)

        left = np.clip(left, -max_wheel_vel, max_wheel_vel)
        right = np.clip(right, -max_wheel_vel, max_wheel_vel)

        return left, right

    def get_robot_velocity_from_wheels(self, left_vel: float, right_vel: float) -> Tuple[float, float]:
        """Convenience method for forward kinematics."""
        return self.forward_kinematics(left_vel, right_vel)

    def get_wheel_velocities_from_robot(self, v: float, omega: float) -> Tuple[float, float]:
        """Convenience method for inverse kinematics."""
        return self.inverse_kinematics(v, omega)


class KinematicController:
    """
    High-level kinematic controller for path following.

    Converts path waypoints into velocity commands using pure pursuit
    or similar geometric tracking.
    """

    def __init__(self, kinematics: DifferentialDriveKinematics,
                 lookahead_distance: float = 0.3):
        self.kinematics = kinematics
        self.lookahead = lookahead_distance

    def pure_pursuit(self, state: RobotState,
                     path: list[Tuple[float, float]],
                     closest_idx: int = 0) -> Tuple[float, float, int]:
        """
        Pure pursuit path tracking controller.

        Args:
            state: Current robot state
            path: List of (x, y) waypoints
            closest_idx: Index of closest waypoint on path

        Returns:
            (v, omega, updated_closest_idx)
        """
        if not path or closest_idx >= len(path):
            return 0.0, 0.0, closest_idx

        # Find lookahead point
        robot_pos = np.array([state.x, state.y])
        lookahead_idx = closest_idx

        for i in range(closest_idx, len(path)):
            dist = np.linalg.norm(np.array(path[i]) - robot_pos)
            if dist >= self.lookahead:
                lookahead_idx = i
                break
        else:
            lookahead_idx = len(path) - 1

        # Update closest index
        min_dist = float('inf')
        new_closest = closest_idx
        for i in range(closest_idx, min(closest_idx + 10, len(path))):
            dist = np.linalg.norm(np.array(path[i]) - robot_pos)
            if dist < min_dist:
                min_dist = dist
                new_closest = i

        # Compute curvature to lookahead point
        target = np.array(path[lookahead_idx])
        robot_to_target = target - robot_pos

        # Transform to robot frame
        cos_theta = np.cos(state.theta)
        sin_theta = np.sin(state.theta)
        local_x = cos_theta * robot_to_target[0] + sin_theta * robot_to_target[1]
        local_y = -sin_theta * robot_to_target[0] + cos_theta * robot_to_target[1]

        if abs(local_x) < 1e-6:
            return 0.0, 0.0, new_closest

        # Curvature = 2 * y / L^2 where L is lookahead distance
        L_sq = local_x**2 + local_y**2
        curvature = 2 * local_y / L_sq if L_sq > 1e-6 else 0.0

        # Convert to velocities
        v = min(self.kinematics.config.max_linear_velocity, 0.5)  # Constant speed for now
        omega = v * curvature

        # Clamp omega
        omega = np.clip(omega, -self.kinematics.config.max_angular_velocity,
                        self.kinematics.config.max_angular_velocity)

        return v, omega, new_closest


def simulate_trajectory(initial_state: RobotState,
                        velocity_profile: list[Tuple[float, float, float]],
                        kinematics: DifferentialDriveKinematics) -> list[RobotState]:
    """
    Simulate robot trajectory given velocity profile.

    Args:
        initial_state: Starting robot state
        velocity_profile: List of (v, omega, dt) commands
        kinematics: Kinematics model

    Returns:
        List of robot states over time
    """
    states = [initial_state]
    current = initial_state

    for v, omega, dt in velocity_profile:
        current = kinematics.integrate_velocity(current, v, omega, dt)
        states.append(current)

    return states


if __name__ == "__main__":
    # Test kinematics
    config = RobotConfig()
    kin = DifferentialDriveKinematics(config)

    # Test forward kinematics
    left, right = 10.0, 10.0  # rad/s
    v, omega = kin.forward_kinematics(left, right)
    print(f"Forward: left={left}, right={right} -> v={v:.4f}, ω={omega:.4f}")

    # Test inverse kinematics
    v, omega = 0.1, 0.5
    left, right = kin.inverse_kinematics(v, omega)
    print(f"Inverse: v={v}, ω={omega} -> left={left:.4f}, right={right:.4f}")

    # Test integration
    state = RobotState(x=0, y=0, theta=0)
    for _ in range(10):
        state = kin.integrate_velocity(state, 0.1, 0.5, 0.1)
        print(f"State: x={state.x:.3f}, y={state.y:.3f}, θ={state.theta:.3f}")

    # Test pure pursuit
    controller = KinematicController(kin)
    path = [(0, 0), (1, 0), (1, 1), (0, 1)]
    state = RobotState(x=0.1, y=0.1, theta=0)
    v, omega, idx = controller.pure_pursuit(state, path, 0)
    print(f"Pure pursuit: v={v:.3f}, ω={omega:.3f}, idx={idx}")