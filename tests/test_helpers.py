"""
Tests for shared helper utilities (python_analysis/utils/helpers.py).
"""

import numpy as np

from helpers import (
    PIDController,
    angle_difference,
    bresenham_line,
    compute_tracking_error,
    create_occupancy_grid,
    euclidean_distance,
    grid_to_world,
    log_odds_to_probability,
    manhattan_distance,
    normalize_angle,
    probability_to_log_odds,
    update_occupancy_grid,
    world_to_grid,
)


class TestCoordinateTransforms:
    def test_world_to_grid_roundtrip(self):
        row, col = world_to_grid((5.0, 5.0), 0.1, (0.0, 0.0))
        x, y = grid_to_world((row, col), 0.1, (0.0, 0.0))
        assert abs(x - 5.0) <= 0.1
        assert abs(y - 5.0) <= 0.1

    def test_world_to_grid_origin_offset(self):
        # Cell (0,0) starts at world (1, 1)
        row, col = world_to_grid((1.0, 1.0), 0.5, (1.0, 1.0))
        assert (row, col) == (0, 0)

    def test_grid_axes_convention(self):
        # x maps to columns, y maps to rows
        row, col = world_to_grid((2.0, 1.0), 1.0, (0.0, 0.0))
        assert (row, col) == (1, 2)


class TestAngles:
    def test_normalize_angle(self):
        assert abs(normalize_angle(3 * np.pi) - np.pi) < 1e-9
        assert abs(normalize_angle(-3 * np.pi) + np.pi) < 1e-9
        assert abs(normalize_angle(0.5) - 0.5) < 1e-9

    def test_angle_difference_wraparound(self):
        # Shortest path from -pi/2 to +pi/2 is +pi (not -3pi)
        diff = angle_difference(np.pi / 2, -np.pi / 2)
        assert abs(abs(diff) - np.pi) < 1e-9


class TestDistances:
    def test_euclidean(self):
        assert euclidean_distance((0, 0), (3, 4)) == 5.0

    def test_manhattan(self):
        assert manhattan_distance((0, 0), (3, 4)) == 7.0


class TestOccupancyGrid:
    def test_log_odds_roundtrip(self):
        probs = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        recovered = log_odds_to_probability(probability_to_log_odds(probs))
        assert np.allclose(probs, recovered, atol=1e-5)

    def test_create_default_unknown(self):
        grid = create_occupancy_grid(10, 10, 0.5)
        assert grid.shape == (10, 10)
        assert np.all(grid == 0.0)  # log-odds 0 == probability 0.5

    def test_update_marks_hit_and_miss(self):
        grid = create_occupancy_grid(10, 10, 0.5)
        grid = update_occupancy_grid(grid, hits=[(5, 5)], misses=[(5, 4), (5, 3)])
        assert grid[5, 5] > 0.0  # more likely occupied
        assert grid[5, 4] < 0.0  # more likely free


class TestBresenham:
    def test_endpoints_included(self):
        cells = bresenham_line(0, 0, 4, 4)
        assert cells[0] == (0, 0)
        assert cells[-1] == (4, 4)
        assert len(cells) == 5

    def test_straight_line(self):
        # bresenham_line(x0, y0, x1, y1) returns (row, col) = (y, x) cells
        cells = bresenham_line(0, 0, 5, 0)  # horizontal in x
        assert all(r == 0 for r, c in cells)
        assert [c for r, c in cells] == [0, 1, 2, 3, 4, 5]


class TestPIDController:
    def test_proportional_response(self):
        pid = PIDController(kp=2.0, ki=0.0, kd=0.0, output_limits=(-10, 10))
        assert pid.step(1.0, 0.1) == 2.0

    def test_integral_accumulates(self):
        pid = PIDController(kp=0.0, ki=1.0, kd=0.0, output_limits=(-10, 10))
        out1 = pid.step(1.0, 0.1)
        out2 = pid.step(1.0, 0.1)
        assert out1 == 0.1
        assert out2 == 0.2  # integral of 1.0 over 0.2 s

    def test_output_limits_enforced(self):
        pid = PIDController(kp=100.0, ki=0.0, kd=0.0, output_limits=(-1, 1))
        assert pid.step(5.0, 0.1) == 1.0

    def test_zero_dt_safe(self):
        pid = PIDController(1.0, 1.0, 1.0)
        assert pid.step(1.0, 0.0) == 0.0

    def test_reset(self):
        pid = PIDController(kp=0.0, ki=1.0, kd=0.0)
        pid.step(1.0, 0.1)
        pid.reset()
        assert pid.integral == 0.0
        assert pid.first_step is True


class TestTrackingError:
    def test_perfect_tracking(self):
        path = [(0, 0), (1, 0), (2, 0)]
        metrics = compute_tracking_error(path, path)
        assert metrics["max_error"] == 0.0
        assert metrics["rmse"] == 0.0

    def test_offset_tracking(self):
        path = [(0, 0), (1, 0), (2, 0)]
        traj = [(0, 1), (1, 1), (2, 1)]
        metrics = compute_tracking_error(path, traj)
        assert abs(metrics["mean_error"] - 1.0) < 1e-9
