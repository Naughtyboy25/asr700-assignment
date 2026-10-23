"""
Tests for Component A: perception, planning (Dijkstra/A*), and control.

Covers both the offline reproducible simulation
(python_simulations/component_a/run_component_a.py) and the Webots
controller modules (webots_project/controllers/robot_controller_a/).
"""

import importlib.util
import os

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_module(name, rel_path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sim_a = _load_module("run_component_a", "python_simulations/component_a/run_component_a.py")

from control import DualPIDController, PIDGains  # noqa: E402
from kinematics import DifferentialDriveKinematics, RobotState  # noqa: E402
from planning import OccupancyGridMap, astar, compare_planners, dijkstra  # noqa: E402


# ---------------------------------------------------------------------------
# Offline simulation: environment and perception
# ---------------------------------------------------------------------------

class TestEnvironment:
    def test_grid_shape(self):
        grid = sim_a.build_grid()
        assert grid.shape == (sim_a.GS, sim_a.GS)

    def test_obstacles_marked(self):
        grid = sim_a.build_grid()
        assert grid.sum() > 0

    def test_start_goal_free(self):
        grid = sim_a.build_grid()
        sr, sc = int(sim_a.START[1] / sim_a.CELL), int(sim_a.START[0] / sim_a.CELL)
        gr, gc = int(sim_a.GOAL[1] / sim_a.CELL), int(sim_a.GOAL[0] / sim_a.CELL)
        assert grid[sr, sc] == 0
        assert grid[gr, gc] == 0

    def test_lidar_detects_obstacle(self):
        # Beam aimed at the first obstacle must return less than max range
        pose = (*sim_a.START, 0.0)
        angs, ranges = sim_a.lidar_scan(pose)
        assert ranges.min() < 4.0
        assert (ranges > 0).all()

    def test_fusion_output(self):
        pose = (*sim_a.START, 0.0)
        angs, ranges = sim_a.lidar_scan(pose)
        img = sim_a.camera_img(pose)
        fused = sim_a.fuse(angs, ranges, img, pose)
        assert fused.shape == (sim_a.GS, sim_a.GS)
        assert fused.min() >= 0.0 and fused.max() <= 1.0


# ---------------------------------------------------------------------------
# Offline simulation: path planning
# ---------------------------------------------------------------------------

class TestOfflinePlanners:
    @pytest.fixture()
    def paths(self):
        grid = sim_a.build_grid()
        sc = (int(sim_a.START[1] / sim_a.CELL), int(sim_a.START[0] / sim_a.CELL))
        gc = (int(sim_a.GOAL[1] / sim_a.CELL), int(sim_a.GOAL[0] / sim_a.CELL))
        pd, ed = sim_a.dijkstra(grid, sc, gc)
        pa, ea = sim_a.astar(grid, sc, gc)
        return grid, sc, gc, pd, ed, pa, ea

    def test_both_find_path(self, paths):
        _, sc, gc, pd, _, pa, _ = paths
        assert pd[0] == sc and pd[-1] == gc
        assert pa[0] == sc and pa[-1] == gc

    def test_paths_are_optimal_and_equal(self, paths):
        _, _, _, pd, _, pa, _ = paths
        # Both algorithms are optimal on the same grid -> same cell count
        assert len(pd) == len(pa)

    def test_astar_explores_no_more_than_dijkstra(self, paths):
        _, _, _, _, ed, _, ea = paths
        assert ea <= ed

    def test_path_avoids_obstacles(self, paths):
        grid, _, _, pd, _, pa, _ = paths
        for r, c in pd + pa:
            assert grid[r, c] == 0

    def test_path_moves_between_adjacent_cells(self, paths):
        _, _, _, pd, _, pa, _ = paths
        for path in (pd, pa):
            for (r1, c1), (r2, c2) in zip(path, path[1:]):
                assert max(abs(r2 - r1), abs(c2 - c1)) == 1


# ---------------------------------------------------------------------------
# Offline simulation: closed-loop PID tracking
# ---------------------------------------------------------------------------

class TestOfflineTracking:
    def test_robot_reaches_goal(self):
        grid = sim_a.build_grid()
        gc = (int(sim_a.GOAL[1] / sim_a.CELL), int(sim_a.GOAL[0] / sim_a.CELL))
        sc = (int(sim_a.START[1] / sim_a.CELL), int(sim_a.START[0] / sim_a.CELL))
        pa, _ = sim_a.astar(grid, sc, gc)
        wa = sim_a.to_world(pa)
        robot, errors = sim_a.track(wa)
        final = robot.traj[-1]
        dist = np.hypot(final[0] - sim_a.GOAL[0], final[1] - sim_a.GOAL[1])
        assert dist < 0.3

    def test_heading_error_decreases(self):
        grid = sim_a.build_grid()
        gc = (int(sim_a.GOAL[1] / sim_a.CELL), int(sim_a.GOAL[0] / sim_a.CELL))
        sc = (int(sim_a.START[1] / sim_a.CELL), int(sim_a.START[0] / sim_a.CELL))
        pa, _ = sim_a.astar(grid, sc, gc)
        wa = sim_a.to_world(pa)
        _, errors = sim_a.track(wa)
        err = np.array([e[1] for e in errors])
        assert np.mean(np.abs(err[:20])) > np.mean(np.abs(err[-20:]))


# ---------------------------------------------------------------------------
# Webots controller modules (pure logic, no Webots runtime needed)
# ---------------------------------------------------------------------------

def _small_grid():
    grid = np.zeros((20, 20), dtype=np.float32)
    grid[8:12, 8:12] = 1.0  # central obstacle block
    return OccupancyGridMap(grid, resolution=0.05)


class TestWebotsPlanners:
    def test_dijkstra_finds_shortest_path(self):
        gm = _small_grid()
        path, stats = dijkstra(gm, (2, 2), (17, 17))
        assert path[0] == (2, 2)
        assert path[-1] == (17, 17)
        assert stats.path_length > 0
        for r, c in path:
            assert gm.is_free(r, c)

    def test_astar_matches_dijkstra_cost(self):
        gm = _small_grid()
        p_d, s_d = dijkstra(gm, (2, 2), (17, 17))
        p_a, s_a = astar(gm, (2, 2), (17, 17))
        assert abs(s_d.path_length - s_a.path_length) < 1e-6
        assert s_a.nodes_explored <= s_d.nodes_explored

    def test_compare_planners(self):
        gm = _small_grid()
        result = compare_planners(gm, (2, 2), (17, 17))
        assert "dijkstra" in result and "astar" in result

    def test_no_path_when_goal_blocked(self):
        gm = _small_grid()
        path, _ = dijkstra(gm, (2, 2), (10, 10))  # inside obstacle
        assert path == []


class TestKinematics:
    def test_forward_straight_line(self):
        kin = DifferentialDriveKinematics()
        v, omega = kin.forward_kinematics(5.0, 5.0)
        assert omega == 0.0
        assert v == pytest.approx(kin.R * 5.0)

    def test_turn_in_place(self):
        kin = DifferentialDriveKinematics()
        v, omega = kin.forward_kinematics(-5.0, 5.0)
        assert v == pytest.approx(0.0)
        assert omega > 0

    def test_inverse_forward_roundtrip(self):
        kin = DifferentialDriveKinematics()
        for v, w in [(0.1, 0.0), (0.0, 1.0), (0.08, -0.5)]:
            left, right = kin.inverse_kinematics(v, w)
            v2, w2 = kin.forward_kinematics(left, right)
            assert v2 == pytest.approx(v)
            assert w2 == pytest.approx(w)


class TestDualPID:
    def test_set_gains_applies_ki_correctly(self):
        # Regression test: set_gains used to copy kd into the integral gain
        ctrl = DualPIDController()
        gains = PIDGains(kp=1.0, ki=2.0, kd=3.0)
        ctrl.set_gains(angular_gains=gains)
        assert ctrl.angular_pid.kp == 1.0
        assert ctrl.angular_pid.ki == 2.0
        assert ctrl.angular_pid.kd == 3.0

    def test_compute_returns_bounded_commands(self):
        ctrl = DualPIDController()
        state = RobotState(x=0.0, y=0.0, theta=0.0)
        v, w = ctrl.compute(state, (1.0, 1.0), dt=0.05)
        assert abs(v) <= ctrl.robot_config.max_linear_velocity
        assert abs(w) <= ctrl.robot_config.max_angular_velocity
