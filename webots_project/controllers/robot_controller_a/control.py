"""
PID Control System for Component A.

Implements PID controllers for differential drive robot path tracking
with gain tuning and response analysis utilities.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from python_analysis.utils.config import PIDConfig, RobotConfig
from python_analysis.utils.helpers import PIDController, normalize_angle, angle_difference
from webots_project.controllers.robot_controller_a.kinematics import DifferentialDriveKinematics, RobotState


@dataclass
class PIDGains:
    """PID gain set for a single control loop."""
    kp: float
    ki: float
    kd: float


@dataclass
class ControlResponse:
    """Control system response data for analysis."""
    time: List[float] = field(default_factory=list)
    setpoint: List[float] = field(default_factory=list)
    measured: List[float] = field(default_factory=list)
    error: List[float] = field(default_factory=list)
    output: List[float] = field(default_factory=list)


class DualPIDController:
    """
    Dual PID controller for differential drive robot.

    Controls both linear velocity (distance to waypoint) and
    angular velocity (heading to waypoint).
    """

    def __init__(self, config: PIDConfig = None, robot_config: RobotConfig = None):
        self.config = config or PIDConfig()
        self.robot_config = robot_config or RobotConfig()

        # Linear velocity PID (distance control)
        self.linear_pid = PIDController(
            kp=self.config.kp_linear,
            ki=self.config.ki_linear,
            kd=self.config.kd_linear,
            output_limits=(-self.robot_config.max_linear_velocity,
                          self.robot_config.max_linear_velocity),
            integral_limits=(-5.0, 5.0)
        )

        # Angular velocity PID (heading control)
        self.angular_pid = PIDController(
            kp=self.config.kp_angular,
            ki=self.config.ki_angular,
            kd=self.config.kd_angular,
            output_limits=(-self.robot_config.max_angular_velocity,
                          self.robot_config.max_angular_velocity),
            integral_limits=(-10.0, 10.0)
        )

        # Response recording
        self.linear_response = ControlResponse()
        self.angular_response = ControlResponse()
        self._recording = False
        self._start_time = 0.0

    def start_recording(self):
        """Start recording control response data."""
        self.linear_response = ControlResponse()
        self.angular_response = ControlResponse()
        self._recording = True
        self._start_time = 0.0

    def stop_recording(self):
        """Stop recording control response data."""
        self._recording = False

    def reset(self):
        """Reset both PID controllers."""
        self.linear_pid.reset()
        self.angular_pid.reset()

    def set_gains(self, linear_gains: PIDGains = None, angular_gains: PIDGains = None):
        """Update PID gains."""
        if linear_gains:
            self.linear_pid.kp = linear_gains.kp
            self.linear_pid.ki = linear_gains.ki
            self.linear_pid.kd = linear_gains.kd
        if angular_gains:
            self.angular_pid.kp = angular_gains.kp
            self.angular_pid.ki = angular_gains.kd
            self.angular_pid.kd = angular_gains.kd

    def compute(self, state: RobotState, target: Tuple[float, float],
                dt: float, current_time: float = 0.0) -> Tuple[float, float]:
        """
        Compute velocity commands to reach target.

        Args:
            state: Current robot state
            target: Target (x, y) position
            dt: Time step (s)
            current_time: Current simulation time (for recording)

        Returns:
            (linear_velocity, angular_velocity) commands
        """
        # Distance to target
        dx = target[0] - state.x
        dy = target[1] - state.y
        distance = np.hypot(dx, dy)

        # Desired heading
        desired_heading = np.arctan2(dy, dx)
        heading_error = angle_difference(desired_heading, state.theta)

        # Linear velocity control (proportional to distance)
        linear_error = distance
        v = self.linear_pid.step(linear_error, dt)

        # Angular velocity control (heading error)
        omega = self.angular_pid.step(heading_error, dt)

        # If very close to target, stop
        if distance < 0.05:
            v = 0.0
            omega = 0.0

        # Record response
        if self._recording:
            t = current_time - self._start_time if self._start_time else current_time
            self.linear_response.time.append(t)
            self.linear_response.setpoint.append(0.0)  # Target distance = 0
            self.linear_response.measured.append(distance)
            self.linear_response.error.append(linear_error)
            self.linear_response.output.append(v)

            self.angular_response.time.append(t)
            self.angular_response.setpoint.append(desired_heading)
            self.angular_response.measured.append(state.theta)
            self.angular_response.error.append(heading_error)
            self.angular_response.output.append(omega)

        return v, omega

    def get_response_data(self) -> Dict[str, ControlResponse]:
        """Get recorded response data."""
        return {
            'linear': self.linear_response,
            'angular': self.angular_response
        }


class PathTrackingController:
    """
    High-level path tracking controller.

    Follows a sequence of waypoints using the dual PID controller.
    """

    def __init__(self, kinematics: DifferentialDriveKinematics,
                 pid_config: PIDConfig = None,
                 waypoint_tolerance: float = 0.15,
                 lookahead_distance: float = 0.3):
        self.kinematics = kinematics
        self.pid_controller = DualPIDController(pid_config, kinematics.config)
        self.waypoint_tolerance = waypoint_tolerance
        self.lookahead_distance = lookahead_distance
        self.current_waypoint_idx = 0
        self.path: List[Tuple[float, float]] = []

    def set_path(self, path: List[Tuple[float, float]]):
        """Set new path to follow."""
        self.path = path
        self.current_waypoint_idx = 0
        self.pid_controller.reset()

    def get_current_target(self, state: RobotState) -> Tuple[float, float]:
        """
        Get current target waypoint using lookahead.

        Args:
            state: Current robot state

        Returns:
            Target (x, y) position
        """
        if not self.path:
            return state.x, state.y

        # Find closest waypoint
        robot_pos = np.array([state.x, state.y])
        min_dist = float('inf')
        closest_idx = self.current_waypoint_idx

        for i in range(self.current_waypoint_idx, len(self.path)):
            dist = np.linalg.norm(np.array(self.path[i]) - robot_pos)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        self.current_waypoint_idx = closest_idx

        # Find lookahead point
        target_idx = closest_idx
        for i in range(closest_idx, len(self.path)):
            dist = np.linalg.norm(np.array(self.path[i]) - robot_pos)
            if dist >= self.lookahead_distance:
                target_idx = i
                break
        else:
            target_idx = len(self.path) - 1

        return self.path[target_idx]

    def is_path_complete(self, state: RobotState) -> bool:
        """Check if final waypoint reached."""
        if not self.path:
            return True
        final_wp = np.array(self.path[-1])
        robot_pos = np.array([state.x, state.y])
        return np.linalg.norm(final_wp - robot_pos) < self.waypoint_tolerance

    def step(self, state: RobotState, dt: float, current_time: float = 0.0) -> Tuple[float, float, bool]:
        """
        Execute one control step.

        Args:
            state: Current robot state
            dt: Time step (s)
            current_time: Current simulation time

        Returns:
            (v, omega, path_complete)
        """
        if self.is_path_complete(state):
            return 0.0, 0.0, True

        target = self.get_current_target(state)
        v, omega = self.pid_controller.compute(state, target, dt, current_time)
        complete = self.is_path_complete(state)

        return v, omega, complete

    def get_response_data(self) -> Dict[str, ControlResponse]:
        """Get PID response data."""
        return self.pid_controller.get_response_data()


def analyze_step_response(response: ControlResponse) -> Dict[str, float]:
    """
    Analyze step response characteristics.

    Args:
        response: ControlResponse with step response data

    Returns:
        Dictionary with overshoot, rise_time, settling_time, steady_state_error
    """
    if not response.time or len(response.time) < 2:
        return {"overshoot": 0.0, "rise_time": 0.0,
                "settling_time": 0.0, "steady_state_error": 0.0}

    time = np.array(response.time)
    error = np.array(response.error)
    output = np.array(response.output)
    setpoint = np.array(response.setpoint)

    # Assume step input at t=0, steady state is last 20% of data
    n_steady = max(1, len(time) // 5)
    steady_state = np.mean(error[-n_steady:]) if n_steady > 0 else error[-1]
    steady_state_output = np.mean(output[-n_steady:]) if n_steady > 0 else output[-1]

    # Overshoot (for error: how much it goes past zero)
    max_error = np.max(np.abs(error))
    initial_error = np.abs(error[0]) if len(error) > 0 else 0
    overshoot = max_error / initial_error - 1.0 if initial_error > 0 else 0.0
    overshoot = max(0.0, overshoot)  # Only positive overshoot

    # Rise time: time to reach 90% of steady state (for error: decay to 10%)
    # For error response, rise time = time to reach 10% of initial error
    target_error = 0.1 * initial_error
    rise_idx = np.where(np.abs(error) <= target_error)[0]
    rise_time = time[rise_idx[0]] if len(rise_idx) > 0 else time[-1]

    # Settling time: time to stay within 2% of steady state
    settle_band = 0.02 * initial_error
    settled = False
    settling_time = time[-1]
    for i in range(len(error)):
        if all(np.abs(error[j]) <= settle_band for j in range(i, len(error))):
            settling_time = time[i]
            settled = True
            break

    # Steady state error
    steady_state_error = np.abs(steady_state)

    return {
        "overshoot": float(overshoot),
        "rise_time": float(rise_time),
        "settling_time": float(settling_time),
        "steady_state_error": float(steady_state_error),
        "steady_state_output": float(steady_state_output)
    }


def tune_pid_ziegler_nichols(plant_response: callable,
                             max_ku: float = 100.0,
                             step_size: float = 0.1) -> Tuple[float, float]:
    """
    Ziegler-Nichols tuning method (approximate for simulation).

    Args:
        plant_response: Function that simulates plant with given Kp
        max_ku: Maximum ultimate gain to search
        step_size: Step size for gain search

    Returns:
        (Ku, Tu) - Ultimate gain and period
    """
    # This is a simplified version - in practice you'd oscillate the system
    # For the assignment, manual tuning with response analysis is expected
    ku = max_ku * 0.6  # Placeholder
    tu = 2.0  # Placeholder
    return ku, tu


def compute_ziegler_nichols_gains(ku: float, tu: float, controller_type: str = "PID") -> PIDGains:
    """
    Compute PID gains using Ziegler-Nichols rules.

    Args:
        ku: Ultimate gain
        tu: Ultimate period
        controller_type: "P", "PI", or "PID"

    Returns:
        PIDGains
    """
    if controller_type == "P":
        return PIDGains(kp=0.5*ku, ki=0.0, kd=0.0)
    elif controller_type == "PI":
        return PIDGains(kp=0.45*ku, ki=0.54*ku/tu, kd=0.0)
    elif controller_type == "PID":
        return PIDGains(kp=0.6*ku, ki=1.2*ku/tu, kd=0.075*ku*tu)
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")


def simulate_closed_loop(controller: DualPIDController,
                         kinematics: DifferentialDriveKinematics,
                         initial_state: RobotState,
                         target: Tuple[float, float],
                         dt: float = 0.1,
                         max_time: float = 30.0) -> List[RobotState]:
    """
    Simulate closed-loop response for analysis.

    Args:
        controller: PID controller
        kinematics: Robot kinematics
        initial_state: Starting state
        target: Target position
        dt: Simulation time step
        max_time: Maximum simulation time

    Returns:
        List of robot states over time
    """
    controller.start_recording()
    controller._start_time = 0.0

    state = initial_state
    states = [state]
    t = 0.0

    while t < max_time:
        v, omega = controller.compute(state, target, dt, t)
        state = kinematics.integrate_velocity(state, v, omega, dt)
        states.append(state)
        t += dt

        # Check if reached target
        dist = np.hypot(target[0] - state.x, target[1] - state.y)
        if dist < 0.05:
            break

    controller.stop_recording()
    return states


if __name__ == "__main__":
    # Test control system
    from webots_project.controllers.robot_controller_a.kinematics import RobotState

    config = PIDConfig()
    robot_config = RobotConfig()
    kin = DifferentialDriveKinematics(robot_config)
    controller = DualPIDController(config, robot_config)

    # Simulate step response
    initial = RobotState(x=0, y=0, theta=0)
    target = (2.0, 0.0)

    states = simulate_closed_loop(controller, kin, initial, target, dt=0.05, max_time=20.0)

    # Analyze response
    responses = controller.get_response_data()
    linear_analysis = analyze_step_response(responses['linear'])
    angular_analysis = analyze_step_response(responses['angular'])

    print("Linear Response Analysis:")
    for k, v in linear_analysis.items():
        print(f"  {k}: {v:.4f}")

    print("\nAngular Response Analysis:")
    for k, v in angular_analysis.items():
        print(f"  {k}: {v:.4f}")

    print(f"\nFinal position: ({states[-1].x:.3f}, {states[-1].y:.3f})")
    print(f"Target: {target}")
    print(f"Distance to target: {np.hypot(target[0]-states[-1].x, target[1]-states[-1].y):.4f}")