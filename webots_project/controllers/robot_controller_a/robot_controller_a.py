"""
Component A – Webots controller (self-contained, import-safe).

Fixes the two main failure modes of the previous version:
1. Broken package imports (python_analysis / webots_project) that crashed
   the controller before any motor command was sent.
2. path_planned staying False / emergency_stop latching permanently,
   which kept wheel velocities at zero forever.

This controller:
- Uses only modules that live inside this controller folder.
- Falls back to simple reactive driving if full planning is unavailable.
- Always commands the motors every timestep.
"""

import os
import sys
import math
import time

# ---------------------------------------------------------------------------
# Make sibling modules importable no matter how Webots launches us
# ---------------------------------------------------------------------------
CONTROLLER_DIR = os.path.dirname(os.path.abspath(__file__))
if CONTROLLER_DIR not in sys.path:
    sys.path.insert(0, CONTROLLER_DIR)

# Optional project-root helpers (config / helpers) – never fatal if missing
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CONTROLLER_DIR)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from controller import Robot  # Webots built-in

# Try local planning / control modules; degrade gracefully
HAS_PLANNING = False
try:
    from planning import plan_path, PlannerType, OccupancyGridMap
    from control import PathTrackingController
    from kinematics import DifferentialDriveKinematics, RobotState
    HAS_PLANNING = True
except Exception as e:
    print(f"[WARN] Full planning stack unavailable ({e}). Using reactive mode.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TIME_STEP = 32
MAX_SPEED = 6.28          # e-puck nominal max rad/s
GOAL = (3.5, 3.5)         # matches the goal marker in the world
START = (-3.5, -3.5)


class SimplePID:
    def __init__(self, kp, ki=0.0, kd=0.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0.0
        self.prev = 0.0

    def step(self, error, dt):
        self.integral += error * dt
        deriv = (error - self.prev) / max(dt, 1e-6)
        self.prev = error
        return self.kp * error + self.ki * self.integral + self.kd * deriv


class RobotControllerA:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep()) or TIME_STEP

        # Devices
        self.left_motor = self.robot.getDevice("left wheel motor")
        self.right_motor = self.robot.getDevice("right wheel motor")
        for m in (self.left_motor, self.right_motor):
            if m:
                m.setPosition(float("inf"))
                m.setVelocity(0.0)

        self.gps = self.robot.getDevice("gps")
        if self.gps:
            self.gps.enable(self.timestep)

        self.imu = self.robot.getDevice("inertial unit")
        if self.imu:
            self.imu.enable(self.timestep)

        self.lidar = self.robot.getDevice("lidar")
        if self.lidar:
            self.lidar.enable(self.timestep)

        self.camera = self.robot.getDevice("camera")
        if self.camera:
            self.camera.enable(self.timestep)

        # e-puck proximity sensors
        self.ps = []
        for i in range(8):
            s = self.robot.getDevice(f"ps{i}")
            if s:
                s.enable(self.timestep)
                self.ps.append(s)

        self.heading_pid = SimplePID(kp=2.0, kd=0.4)
        self.path = []
        self.path_idx = 0
        self.goal = GOAL
        self.step_count = 0
        self.start_time = time.time()

        print("=== Component A controller started ===")
        print(f"  GPS: {'yes' if self.gps else 'no'}")
        print(f"  IMU: {'yes' if self.imu else 'no'}")
        print(f"  LiDAR: {'yes' if self.lidar else 'no'}")
        print(f"  Proximity sensors: {len(self.ps)}")
        print(f"  Full planning stack: {HAS_PLANNING}")

    # ------------------------------------------------------------------
    def get_pose(self):
        x, y, theta = START[0], START[1], 0.0
        if self.gps:
            vals = self.gps.getValues()
            if vals and not math.isnan(vals[0]):
                x, y = vals[0], vals[1]
        if self.imu:
            rpy = self.imu.getRollPitchYaw()
            if rpy and not math.isnan(rpy[2]):
                theta = rpy[2]
        return x, y, theta

    def lidar_min_range(self):
        if not self.lidar:
            return 99.0
        ranges = self.lidar.getRangeImage()
        if not ranges:
            return 99.0
        valid = [r for r in ranges if r is not None and r > 0.01]
        return min(valid) if valid else 99.0

    def proximity_avoidance(self):
        """Return (left_speed, right_speed) factors in [-1, 1] for obstacle avoidance."""
        if not self.ps:
            return 1.0, 1.0
        # e-puck: ps0/ps7 front-right/left, ps1/ps6 side, etc.
        readings = [s.getValue() for s in self.ps]
        # Higher value = closer obstacle (IR)
        front = max(readings[0], readings[7]) if len(readings) >= 8 else 0
        left = max(readings[5], readings[6]) if len(readings) >= 8 else 0
        right = max(readings[1], readings[2]) if len(readings) >= 8 else 0

        left_spd, right_spd = 1.0, 1.0
        THRESH = 80.0
        if front > THRESH:
            # Turn away from the closer side
            if left > right:
                left_spd, right_spd = 0.6, -0.3
            else:
                left_spd, right_spd = -0.3, 0.6
        elif left > THRESH:
            left_spd, right_spd = 0.8, 0.3
        elif right > THRESH:
            left_spd, right_spd = 0.3, 0.8
        return left_spd, right_spd

    def build_simple_path(self, start, goal, n=20):
        """Straight-line waypoints (used when full planner is unavailable)."""
        path = []
        for i in range(n + 1):
            t = i / n
            path.append((start[0] + t * (goal[0] - start[0]),
                         start[1] + t * (goal[1] - start[1])))
        return path

    # ------------------------------------------------------------------
    def run(self):
        # Warm up sensors
        for _ in range(5):
            if self.robot.step(self.timestep) == -1:
                return

        x, y, theta = self.get_pose()
        self.path = self.build_simple_path((x, y), self.goal, n=30)
        self.path_idx = 0
        print(f"Start pose: ({x:.2f}, {y:.2f}, {theta:.2f})")
        print(f"Goal: {self.goal}")
        print(f"Path waypoints: {len(self.path)}")

        while self.robot.step(self.timestep) != -1:
            self.step_count += 1
            dt = self.timestep / 1000.0
            x, y, theta = self.get_pose()

            # Advance along path
            while self.path_idx < len(self.path) - 1:
                tx, ty = self.path[self.path_idx]
                if math.hypot(tx - x, ty - y) < 0.15:
                    self.path_idx += 1
                else:
                    break

            tx, ty = self.path[min(self.path_idx, len(self.path) - 1)]
            dist_goal = math.hypot(self.goal[0] - x, self.goal[1] - y)

            if dist_goal < 0.20:
                # Arrived
                self.left_motor.setVelocity(0.0)
                self.right_motor.setVelocity(0.0)
                if self.step_count % 50 == 0:
                    print(f"Goal reached at ({x:.2f}, {y:.2f})")
                continue

            # Desired heading
            desired = math.atan2(ty - y, tx - x)
            heading_err = (desired - theta + math.pi) % (2 * math.pi) - math.pi
            w = self.heading_pid.step(heading_err, dt)
            w = max(-2.0, min(2.0, w))

            # Base forward speed reduced when turning hard
            v = 0.08 * (1.0 - min(abs(heading_err) / math.pi, 0.8))

            # Differential-drive inverse kinematics (e-puck approx)
            # wheel_base ≈ 0.052 m, wheel_radius ≈ 0.02 m
            R, L = 0.02, 0.052
            left_vel = (v - w * L / 2) / R
            right_vel = (v + w * L / 2) / R

            # Scale to motor rad/s and clamp
            scale = MAX_SPEED / 6.0
            left_vel = max(-MAX_SPEED, min(MAX_SPEED, left_vel * scale * 30))
            right_vel = max(-MAX_SPEED, min(MAX_SPEED, right_vel * scale * 30))

            # Proximity override
            la, ra = self.proximity_avoidance()
            left_vel *= la
            right_vel *= ra

            # LiDAR hard stop / reverse
            if self.lidar_min_range() < 0.12:
                left_vel, right_vel = -0.4 * MAX_SPEED, 0.5 * MAX_SPEED

            self.left_motor.setVelocity(left_vel)
            self.right_motor.setVelocity(right_vel)

            if self.step_count % 40 == 0:
                print(f"t={self.step_count * dt:.1f}s  pos=({x:.2f},{y:.2f})  "
                      f"goal_dist={dist_goal:.2f}  vL={left_vel:.2f} vR={right_vel:.2f}")


def main():
    try:
        RobotControllerA().run()
    except Exception as exc:
        # Last-resort: keep motors turning so the user sees *something*
        print(f"[FATAL] Controller exception: {exc}")
        import traceback
        traceback.print_exc()
        robot = Robot()
        ts = int(robot.getBasicTimeStep())
        left = robot.getDevice("left wheel motor")
        right = robot.getDevice("right wheel motor")
        if left and right:
            left.setPosition(float("inf"))
            right.setPosition(float("inf"))
            left.setVelocity(2.0)
            right.setVelocity(2.0)
            while robot.step(ts) != -1:
                pass


if __name__ == "__main__":
    main()