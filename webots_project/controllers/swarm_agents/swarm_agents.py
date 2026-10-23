"""
Webots entry point for swarm agents.
Role is taken from controllerArgs: "leader" or "follower".
"""
import os
import sys
import math
import time

CONTROLLER_DIR = os.path.dirname(os.path.abspath(__file__))
if CONTROLLER_DIR not in sys.path:
    sys.path.insert(0, CONTROLLER_DIR)

from controller import Robot

MAX_SPEED = 6.28
TIME_STEP = 32

# Coverage targets matching the world file
TARGETS = [
    (-3.0, 2.0),
    (3.0, -2.0),
    (2.0, 3.0),
    (-2.0, -3.0),
]


class SwarmAgent:
    def __init__(self, role="follower"):
        self.role = role
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep()) or TIME_STEP
        self.name = self.robot.getName()

        self.left = self.robot.getDevice("left wheel motor")
        self.right = self.robot.getDevice("right wheel motor")
        for m in (self.left, self.right):
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

        self.ps = []
        for i in range(8):
            s = self.robot.getDevice(f"ps{i}")
            if s:
                s.enable(self.timestep)
                self.ps.append(s)

        # Assign target by robot index in name
        idx = 0
        if "follower_1" in self.name:
            idx = 1
        elif "follower_2" in self.name:
            idx = 2
        elif "follower_3" in self.name:
            idx = 3
        self.target = TARGETS[idx % len(TARGETS)]
        self.step_count = 0

        print(f"[{self.name}] role={role} target={self.target}")

    def pose(self):
        x = y = theta = 0.0
        if self.gps:
            v = self.gps.getValues()
            if v and not math.isnan(v[0]):
                x, y = v[0], v[1]
        if self.imu:
            rpy = self.imu.getRollPitchYaw()
            if rpy and not math.isnan(rpy[2]):
                theta = rpy[2]
        return x, y, theta

    def avoid(self):
        if not self.ps:
            return 1.0, 1.0
        r = [s.getValue() for s in self.ps]
        front = max(r[0], r[7]) if len(r) >= 8 else 0
        left = max(r[5], r[6]) if len(r) >= 8 else 0
        right = max(r[1], r[2]) if len(r) >= 8 else 0
        THRESH = 80.0
        if front > THRESH:
            return (0.5, -0.3) if left > right else (-0.3, 0.5)
        if left > THRESH:
            return 0.8, 0.3
        if right > THRESH:
            return 0.3, 0.8
        return 1.0, 1.0

    def run(self):
        for _ in range(5):
            if self.robot.step(self.timestep) == -1:
                return

        while self.robot.step(self.timestep) != -1:
            self.step_count += 1
            x, y, theta = self.pose()
            tx, ty = self.target
            dist = math.hypot(tx - x, ty - y)

            if dist < 0.25:
                # Orbit / hold near target
                self.left.setVelocity(0.8)
                self.right.setVelocity(-0.8)
                if self.step_count % 50 == 0:
                    print(f"[{self.name}] at target ({x:.2f},{y:.2f})")
                continue

            desired = math.atan2(ty - y, tx - x)
            err = (desired - theta + math.pi) % (2 * math.pi) - math.pi
            w = 2.5 * err
            v = 0.07 * (1.0 - min(abs(err) / math.pi, 0.7))

            R, L = 0.02, 0.052
            left_vel = (v - w * L / 2) / R
            right_vel = (v + w * L / 2) / R
            scale = 25.0
            left_vel = max(-MAX_SPEED, min(MAX_SPEED, left_vel * scale))
            right_vel = max(-MAX_SPEED, min(MAX_SPEED, right_vel * scale))

            la, ra = self.avoid()
            left_vel *= la
            right_vel *= ra

            self.left.setVelocity(left_vel)
            self.right.setVelocity(right_vel)

            if self.step_count % 50 == 0:
                print(f"[{self.name}] pos=({x:.2f},{y:.2f}) dist={dist:.2f}")


def main():
    role = sys.argv[1] if len(sys.argv) > 1 else "follower"
    try:
        SwarmAgent(role).run()
    except Exception as e:
        print(f"[FATAL] {e}")
        import traceback
        traceback.print_exc()
        # keep wheels turning so user sees motion
        robot = Robot()
        ts = int(robot.getBasicTimeStep())
        left = robot.getDevice("left wheel motor")
        right = robot.getDevice("right wheel motor")
        if left and right:
            left.setPosition(float("inf"))
            right.setPosition(float("inf"))
            left.setVelocity(1.5)
            right.setVelocity(1.5)
            while robot.step(ts) != -1:
                pass


if __name__ == "__main__":
    main()