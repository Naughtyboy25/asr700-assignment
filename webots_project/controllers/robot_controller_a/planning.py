"""
Motion Planning Algorithms for Component A.

Implements Dijkstra's algorithm and A* algorithm for path planning
on occupancy grids with comparison and visualization utilities.
"""

import heapq
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set, Callable
from enum import Enum
from python_analysis.utils.config import EnvironmentConfig, PathPlanningConfig
from python_analysis.utils.helpers import (world_to_grid, grid_to_world, euclidean_distance,
                                           manhattan_distance, create_occupancy_grid,
                                           probability_to_log_odds, log_odds_to_probability)


class PlannerType(Enum):
    """Path planner types."""
    DIJKSTRA = "dijkstra"
    ASTAR = "astar"


@dataclass
class PlannerStats:
    """Statistics for planner comparison."""
    nodes_explored: int = 0
    path_length: float = 0.0
    computation_time_ms: float = 0.0
    path_found: bool = False
    path: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class GridNode:
    """Node in the search grid."""
    row: int
    col: int
    g_cost: float = float('inf')  # Cost from start
    h_cost: float = 0.0           # Heuristic to goal
    f_cost: float = float('inf')  # Total cost (g + h)
    parent: Optional['GridNode'] = None

    def __lt__(self, other: 'GridNode') -> bool:
        return self.f_cost < other.f_cost

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GridNode):
            return False
        return self.row == other.row and self.col == other.col

    def __hash__(self) -> int:
        return hash((self.row, self.col))


class OccupancyGridMap:
    """
    Occupancy grid map for path planning.

    Handles grid representation, collision checking, and neighbor generation.
    """

    def __init__(self, grid: np.ndarray,
                 resolution: float = 0.05,
                 origin: Tuple[float, float] = (0.0, 0.0),
                 occupancy_threshold: float = 0.5):
        """
        Initialize occupancy grid map.

        Args:
            grid: 2D numpy array with probability values (0=free, 1=occupied)
            resolution: Grid cell size in meters
            origin: World coordinates of grid (0,0)
            occupancy_threshold: Probability threshold for occupied cells
        """
        self.grid = grid
        self.resolution = resolution
        self.origin = origin
        self.threshold = occupancy_threshold
        self.height, self.width = grid.shape

        # Precompute obstacle map
        self.obstacle_map = (grid >= occupancy_threshold).astype(np.uint8)

    def is_valid(self, row: int, col: int) -> bool:
        """Check if cell is within bounds."""
        return 0 <= row < self.height and 0 <= col < self.width

    def is_free(self, row: int, col: int) -> bool:
        """Check if cell is free (not occupied)."""
        if not self.is_valid(row, col):
            return False
        return self.obstacle_map[row, col] == 0

    def is_occupied(self, row: int, col: int) -> bool:
        """Check if cell is occupied."""
        if not self.is_valid(row, col):
            return True  # Out of bounds treated as occupied
        return self.obstacle_map[row, col] == 1

    def get_neighbors(self, row: int, col: int,
                      diagonal: bool = True) -> List[Tuple[int, int]]:
        """
        Get valid neighbor cells.

        Args:
            row, col: Current cell
            diagonal: Allow diagonal movement

        Returns:
            List of valid (row, col) neighbors
        """
        neighbors = []

        # 4-connected
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if self.is_free(nr, nc):
                neighbors.append((nr, nc))

        # 8-connected (diagonal)
        if diagonal:
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = row + dr, col + dc
                if self.is_free(nr, nc):
                    # Check for corner cutting (both adjacent cells must be free)
                    if self.is_free(row + dr, col) and self.is_free(row, col + dc):
                        neighbors.append((nr, nc))

        return neighbors

    def world_to_grid(self, world_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world position to grid coordinates."""
        return world_to_grid(world_pos, self.resolution, self.origin)

    def grid_to_world(self, grid_pos: Tuple[int, int]) -> Tuple[float, float]:
        """Convert grid position to world coordinates."""
        return grid_to_world(grid_pos, self.resolution, self.origin)

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int],
                  heuristic_type: str = "euclidean") -> float:
        """
        Compute heuristic between two grid cells.

        Args:
            a, b: Grid positions (row, col)
            heuristic_type: "euclidean", "manhattan", "octile", "chebyshev"

        Returns:
            Heuristic value
        """
        dr = abs(a[0] - b[0])
        dc = abs(a[1] - b[1])

        if heuristic_type == "euclidean":
            return np.hypot(dr, dc) * self.resolution
        elif heuristic_type == "manhattan":
            return (dr + dc) * self.resolution
        elif heuristic_type == "octile":
            # Octile distance (good for 8-connected grids)
            return (max(dr, dc) + (np.sqrt(2) - 1) * min(dr, dc)) * self.resolution
        elif heuristic_type == "chebyshev":
            return max(dr, dc) * self.resolution
        else:
            return np.hypot(dr, dc) * self.resolution


def reconstruct_path(node: GridNode) -> List[Tuple[int, int]]:
    """Reconstruct path from goal node to start."""
    path = []
    current = node
    while current is not None:
        path.append((current.row, current.col))
        current = current.parent
    path.reverse()
    return path


def dijkstra(grid_map: OccupancyGridMap,
             start: Tuple[int, int],
             goal: Tuple[int, int],
             diagonal: bool = True) -> Tuple[List[Tuple[int, int]], PlannerStats]:
    """
    Dijkstra's algorithm for shortest path.

    Args:
        grid_map: Occupancy grid map
        start: Start (row, col)
        goal: Goal (row, col)
        diagonal: Allow diagonal movement

    Returns:
        (path, stats) - path as list of grid cells, planner statistics
    """
    import time
    start_time = time.perf_counter()

    # Initialize
    start_node = GridNode(start[0], start[1], g_cost=0.0)
    goal_node = GridNode(goal[0], goal[1])

    open_set = [start_node]
    open_dict: Dict[Tuple[int, int], GridNode] = {start: start_node}
    closed_set: Set[Tuple[int, int]] = set()
    nodes_explored = 0

    while open_set:
        current = heapq.heappop(open_set)
        current_pos = (current.row, current.col)

        # Remove from open dict
        open_dict.pop(current_pos, None)

        # Check if goal reached
        if current_pos == goal:
            path = reconstruct_path(current)
            elapsed = (time.perf_counter() - start_time) * 1000

            # Compute path length in world coordinates
            path_length = 0.0
            for i in range(len(path) - 1):
                w1 = grid_map.grid_to_world(path[i])
                w2 = grid_map.grid_to_world(path[i + 1])
                path_length += euclidean_distance(w1, w2)

            stats = PlannerStats(
                nodes_explored=nodes_explored,
                path_length=path_length,
                computation_time_ms=elapsed,
                path_found=True,
                path=path
            )
            return path, stats

        # Add to closed set
        closed_set.add(current_pos)
        nodes_explored += 1

        # Explore neighbors
        for nr, nc in grid_map.get_neighbors(current.row, current.col, diagonal):
            neighbor_pos = (nr, nc)

            if neighbor_pos in closed_set:
                continue

            # Cost to neighbor (Euclidean distance in world coords)
            w1 = grid_map.grid_to_world(current_pos)
            w2 = grid_map.grid_to_world(neighbor_pos)
            step_cost = euclidean_distance(w1, w2)

            new_g_cost = current.g_cost + step_cost

            # Check if neighbor in open set with better cost
            if neighbor_pos in open_dict:
                neighbor = open_dict[neighbor_pos]
                if new_g_cost < neighbor.g_cost:
                    neighbor.g_cost = new_g_cost
                    neighbor.f_cost = new_g_cost
                    neighbor.parent = current
                    heapq.heapify(open_set)  # Re-heapify (simple approach)
            else:
                neighbor = GridNode(nr, nc, g_cost=new_g_cost, f_cost=new_g_cost, parent=current)
                open_dict[neighbor_pos] = neighbor
                heapq.heappush(open_set, neighbor)

    # No path found
    elapsed = (time.perf_counter() - start_time) * 1000
    stats = PlannerStats(
        nodes_explored=nodes_explored,
        path_length=0.0,
        computation_time_ms=elapsed,
        path_found=False,
        path=[]
    )
    return [], stats


def astar(grid_map: OccupancyGridMap,
          start: Tuple[int, int],
          goal: Tuple[int, int],
          diagonal: bool = True,
          heuristic_type: str = "euclidean",
          weight: float = 1.0) -> Tuple[List[Tuple[int, int]], PlannerStats]:
    """
    A* algorithm for shortest path.

    Args:
        grid_map: Occupancy grid map
        start: Start (row, col)
        goal: Goal (row, col)
        diagonal: Allow diagonal movement
        heuristic_type: "euclidean", "manhattan", "octile", "chebyshev"
        weight: Heuristic weight (>=1.0 for weighted A*)

    Returns:
        (path, stats) - path as list of grid cells, planner statistics
    """
    import time
    start_time = time.perf_counter()

    # Initialize
    start_node = GridNode(start[0], start[1], g_cost=0.0)
    goal_pos = goal

    # Compute heuristic for start
    h = grid_map.heuristic(start, goal, heuristic_type)
    start_node.h_cost = h
    start_node.f_cost = weight * h

    goal_node = GridNode(goal[0], goal[1])

    open_set = [start_node]
    open_dict: Dict[Tuple[int, int], GridNode] = {start: start_node}
    closed_set: Set[Tuple[int, int]] = set()
    nodes_explored = 0

    while open_set:
        current = heapq.heappop(open_set)
        current_pos = (current.row, current.col)

        # Remove from open dict
        open_dict.pop(current_pos, None)

        # Check if goal reached
        if current_pos == goal_pos:
            path = reconstruct_path(current)
            elapsed = (time.perf_counter() - start_time) * 1000

            # Compute path length
            path_length = 0.0
            for i in range(len(path) - 1):
                w1 = grid_map.grid_to_world(path[i])
                w2 = grid_map.grid_to_world(path[i + 1])
                path_length += euclidean_distance(w1, w2)

            stats = PlannerStats(
                nodes_explored=nodes_explored,
                path_length=path_length,
                computation_time_ms=elapsed,
                path_found=True,
                path=path
            )
            return path, stats

        # Add to closed set
        closed_set.add(current_pos)
        nodes_explored += 1

        # Explore neighbors
        for nr, nc in grid_map.get_neighbors(current.row, current.col, diagonal):
            neighbor_pos = (nr, nc)

            if neighbor_pos in closed_set:
                continue

            # Cost to neighbor
            w1 = grid_map.grid_to_world(current_pos)
            w2 = grid_map.grid_to_world(neighbor_pos)
            step_cost = euclidean_distance(w1, w2)

            new_g_cost = current.g_cost + step_cost

            # Check if neighbor in open set with better cost
            if neighbor_pos in open_dict:
                neighbor = open_dict[neighbor_pos]
                if new_g_cost < neighbor.g_cost:
                    neighbor.g_cost = new_g_cost
                    neighbor.h_cost = grid_map.heuristic(neighbor_pos, goal, heuristic_type)
                    neighbor.f_cost = new_g_cost + weight * neighbor.h_cost
                    neighbor.parent = current
                    heapq.heapify(open_set)
            else:
                h = grid_map.heuristic(neighbor_pos, goal, heuristic_type)
                neighbor = GridNode(
                    nr, nc,
                    g_cost=new_g_cost,
                    h_cost=h,
                    f_cost=new_g_cost + weight * h,
                    parent=current
                )
                open_dict[neighbor_pos] = neighbor
                heapq.heappush(open_set, neighbor)

    # No path found
    elapsed = (time.perf_counter() - start_time) * 1000
    stats = PlannerStats(
        nodes_explored=nodes_explored,
        path_length=0.0,
        computation_time_ms=elapsed,
        path_found=False,
        path=[]
    )
    return [], stats


def plan_path(grid_map: OccupancyGridMap,
              start_world: Tuple[float, float],
              goal_world: Tuple[float, float],
              planner: PlannerType = PlannerType.ASTAR,
              diagonal: bool = True,
              heuristic_type: str = "euclidean",
              weight: float = 1.0) -> Tuple[List[Tuple[float, float]], PlannerStats]:
    """
    High-level path planning function.

    Args:
        grid_map: Occupancy grid map
        start_world: Start position in world coordinates (x, y)
        goal_world: Goal position in world coordinates (x, y)
        planner: PlannerType.DIJKSTRA or PlannerType.ASTAR
        diagonal: Allow diagonal movement
        heuristic_type: Heuristic type for A*
        weight: Heuristic weight for weighted A*

    Returns:
        (path_world, stats) - path in world coordinates, planner statistics
    """
    # Convert to grid coordinates
    start_grid = grid_map.world_to_grid(start_world)
    goal_grid = grid_map.world_to_grid(goal_world)

    # Check if start/goal are valid
    if not grid_map.is_free(*start_grid):
        # Try to find nearby free cell for start
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = start_grid[0] + dr, start_grid[1] + dc
                if grid_map.is_free(nr, nc):
                    start_grid = (nr, nc)
                    break

    if not grid_map.is_free(*goal_grid):
        # Try to find nearby free cell for goal
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = goal_grid[0] + dr, goal_grid[1] + dc
                if grid_map.is_free(nr, nc):
                    goal_grid = (nr, nc)
                    break

    # Run planner
    if planner == PlannerType.DIJKSTRA:
        path_grid, stats = dijkstra(grid_map, start_grid, goal_grid, diagonal)
    else:
        path_grid, stats = astar(grid_map, start_grid, goal_grid, diagonal, heuristic_type, weight)

    # Convert path to world coordinates
    path_world = [grid_map.grid_to_world(p) for p in path_grid]

    return path_world, stats


def compare_planners(grid_map: OccupancyGridMap,
                     start: Tuple[float, float],
                     goal: Tuple[float, float],
                     runs: int = 5) -> Dict[str, Dict[str, float]]:
    """
    Compare Dijkstra and A* planners over multiple runs.

    Args:
        grid_map: Occupancy grid map
        start: Start position (world coords)
        goal: Goal position (world coords)
        runs: Number of runs for averaging

    Returns:
        Dictionary with averaged statistics for each planner
    """
    results = {
        "dijkstra": {"nodes": [], "length": [], "time": [], "success": []},
        "astar": {"nodes": [], "length": [], "time": [], "success": []}
    }

    for _ in range(runs):
        # Dijkstra
        _, stats_d = plan_path(grid_map, start, goal, PlannerType.DIJKSTRA)
        results["dijkstra"]["nodes"].append(stats_d.nodes_explored)
        results["dijkstra"]["length"].append(stats_d.path_length)
        results["dijkstra"]["time"].append(stats_d.computation_time_ms)
        results["dijkstra"]["success"].append(stats_d.path_found)

        # A*
        _, stats_a = plan_path(grid_map, start, goal, PlannerType.ASTAR)
        results["astar"]["nodes"].append(stats_a.nodes_explored)
        results["astar"]["length"].append(stats_a.path_length)
        results["astar"]["time"].append(stats_a.computation_time_ms)
        results["astar"]["success"].append(stats_a.path_found)

    # Average results
    summary = {}
    for planner in ["dijkstra", "astar"]:
        n = len(results[planner]["nodes"])
        if n > 0:
            summary[planner] = {
                "avg_nodes_explored": np.mean(results[planner]["nodes"]),
                "avg_path_length": np.mean(results[planner]["length"]),
                "avg_computation_time_ms": np.mean(results[planner]["time"]),
                "success_rate": np.mean(results[planner]["success"]),
                "std_nodes": np.std(results[planner]["nodes"]),
                "std_time_ms": np.std(results[planner]["time"])
            }

    return summary


def smooth_path_waypoints(path: List[Tuple[float, float]],
                          grid_map: OccupancyGridMap,
                          max_iterations: int = 100,
                          tolerance: float = 1e-4) -> List[Tuple[float, float]]:
    """
    Smooth path using gradient descent (shortcutting).

    Args:
        path: Path in world coordinates
        grid_map: Occupancy grid for collision checking
        max_iterations: Maximum smoothing iterations
        tolerance: Convergence tolerance

    Returns:
        Smoothed path
    """
    if len(path) < 3:
        return path

    path_arr = np.array(path, dtype=np.float32)
    new_path = path_arr.copy()

    for iteration in range(max_iterations):
        change = 0.0
        for i in range(1, len(path) - 1):
            original = new_path[i].copy()

            # Data term: stay close to original
            new_path[i] += 0.5 * (path_arr[i] - new_path[i])

            # Smoothness term: move toward neighbors
            new_path[i] += 0.1 * (new_path[i-1] + new_path[i+1] - 2*new_path[i])

            # Collision checking: if new point is in collision, revert
            grid_pos = grid_map.world_to_grid(tuple(new_path[i]))
            if not grid_map.is_free(*grid_pos):
                new_path[i] = original

            change += np.linalg.norm(new_path[i] - original)

        if change < tolerance:
            break

    return [tuple(p) for p in new_path]


def shortcut_path(path: List[Tuple[float, float]],
                  grid_map: OccupancyGridMap,
                  max_attempts: int = 1000) -> List[Tuple[float, float]]:
    """
    Shortcut path by removing unnecessary waypoints.

    Args:
        path: Path in world coordinates
        grid_map: Occupancy grid for collision checking
        max_attempts: Maximum shortcut attempts

    Returns:
        Shortcut path
    """
    if len(path) < 3:
        return path

    path = list(path)

    for _ in range(max_attempts):
        if len(path) < 3:
            break

        i = np.random.randint(0, len(path) - 2)
        j = np.random.randint(i + 2, len(path))

        # Check if direct connection is valid
        p1 = np.array(path[i])
        p2 = np.array(path[j])

        # Sample points along line
        num_samples = max(5, int(np.linalg.norm(p2 - p1) / grid_map.resolution))
        collision = False

        for k in range(num_samples + 1):
            t = k / num_samples
            sample = p1 + t * (p2 - p1)
            grid_pos = grid_map.world_to_grid(tuple(sample))
            if not grid_map.is_free(*grid_pos):
                collision = True
                break

        if not collision:
            # Remove intermediate waypoints
            path = path[:i+1] + path[j:]
            break

    return path


def interpolate_path(path: List[Tuple[float, float]],
                     step_size: float = 0.1) -> List[Tuple[float, float]]:
    """
    Interpolate path to have uniform step size between waypoints.

    Args:
        path: Path in world coordinates
        step_size: Desired distance between interpolated points

    Returns:
        Interpolated path
    """
    if len(path) < 2:
        return path

    interpolated = [path[0]]
    for i in range(len(path) - 1):
        p1 = np.array(path[i])
        p2 = np.array(path[i + 1])
        dist = np.linalg.norm(p2 - p1)
        if dist < step_size:
            continue
        num_steps = int(dist / step_size)
        for s in range(1, num_steps + 1):
            t = s / num_steps
            interp = p1 + t * (p2 - p1)
            interpolated.append((float(interp[0]), float(interp[1])))

    # Ensure goal is included
    if np.linalg.norm(np.array(interpolated[-1]) - np.array(path[-1])) > 1e-6:
        interpolated.append(path[-1])

    return interpolated


def is_path_valid(path: List[Tuple[float, float]],
                  grid_map: OccupancyGridMap,
                  check_resolution: float = None) -> bool:
    """
    Check if a path is collision-free.

    Args:
        path: Path in world coordinates
        grid_map: Occupancy grid map
        check_resolution: Distance between collision checks (default: grid resolution)

    Returns:
        True if path is valid (collision-free)
    """
    if check_resolution is None:
        check_resolution = grid_map.resolution

    if len(path) < 2:
        return True

    for i in range(len(path) - 1):
        p1 = np.array(path[i])
        p2 = np.array(path[i + 1])
        dist = np.linalg.norm(p2 - p1)
        num_checks = max(2, int(dist / check_resolution) + 1)

        for k in range(num_checks):
            t = k / (num_checks - 1) if num_checks > 1 else 0
            sample = p1 + t * (p2 - p1)
            grid_pos = grid_map.world_to_grid(tuple(sample))
            if not grid_map.is_free(*grid_pos):
                return False

    return True


def get_path_cells(grid_map: OccupancyGridMap,
                   path: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
    """
    Get grid cells that a path passes through.

    Args:
        grid_map: Occupancy grid map
        path: Path in world coordinates

    Returns:
        List of (row, col) grid cells
    """
    cells = []
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]
        r1, c1 = grid_map.world_to_grid(p1)
        r2, c2 = grid_map.world_to_grid(p2)
        # Bresenham line
        cells.extend(bresenham_line(r1, c1, r2, c2))
    return cells


def bresenham_line(r1: int, c1: int, r2: int, c2: int) -> List[Tuple[int, int]]:
    """Bresenham's line algorithm for grid cells."""
    cells = []
    dr = abs(r2 - r1)
    dc = abs(c2 - c1)
    sr = 1 if r1 < r2 else -1
    sc = 1 if c1 < c2 else -1
    err = dr - dc

    r, c = r1, c1
    while True:
        cells.append((r, c))
        if r == r2 and c == c2:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return cells


def visualize_search(grid_map: OccupancyGridMap,
                     start: Tuple[int, int],
                     goal: Tuple[int, int],
                     open_set: List[Tuple[int, int]] = None,
                     closed_set: Set[Tuple[int, int]] = None,
                     path: List[Tuple[int, int]] = None,
                     save_path: str = None,
                     show: bool = True):
    """
    Visualize search process (explored nodes, open/closed sets, path).

    Args:
        grid_map: Occupancy grid map
        start: Start grid position (row, col)
        goal: Goal grid position (row, col)
        open_set: Set of open nodes (row, col)
        closed_set: Set of closed nodes (row, col)
        path: Final path (row, col)
        save_path: Optional path to save figure
        show: Whether to display the plot
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(10, 10))

    # Occupancy grid
    prob_grid = grid_map.grid
    im = ax.imshow(prob_grid, origin='lower', cmap='RdYlGn_r', vmin=0, vmax=1,
                   extent=[0, grid_map.width * grid_map.resolution,
                           0, grid_map.height * grid_map.resolution],
                   alpha=0.7)

    # Closed set (explored)
    if closed_set:
        closed_rows = [c[0] for c in closed_set]
        closed_cols = [c[1] for c in closed_set]
        ax.scatter([c * grid_map.resolution + grid_map.resolution/2 for c in closed_cols],
                   [r * grid_map.resolution + grid_map.resolution/2 for r in closed_rows],
                   c='gray', s=2, alpha=0.5, label=f'Explored ({len(closed_set)})')

    # Open set (frontier)
    if open_set:
        open_rows = [c[0] for c in open_set]
        open_cols = [c[1] for c in open_set]
        ax.scatter([c * grid_map.resolution + grid_map.resolution/2 for c in open_cols],
                   [r * grid_map.resolution + grid_map.resolution/2 for r in open_rows],
                   c='yellow', s=5, alpha=0.8, label=f'Frontier ({len(open_set)})')

    # Path
    if path:
        path_rows = [c[0] for c in path]
        path_cols = [c[1] for c in path]
        ax.plot([c * grid_map.resolution + grid_map.resolution/2 for c in path_cols],
                [r * grid_map.resolution + grid_map.resolution/2 for r in path_rows],
                'r-', linewidth=3, label=f'Path ({len(path)} steps)')

    # Start/Goal
    ax.scatter(start[1] * grid_map.resolution + grid_map.resolution/2,
               start[0] * grid_map.resolution + grid_map.resolution/2,
               c='green', s=150, marker='o', label='Start', zorder=5)
    ax.scatter(goal[1] * grid_map.resolution + grid_map.resolution/2,
               goal[0] * grid_map.resolution + grid_map.resolution/2,
               c='red', s=150, marker='*', label='Goal', zorder=5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Search Visualization')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='Occupancy Probability')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

    return fig


def visualize_comparison(grid_map: OccupancyGridMap,
                         start_world: Tuple[float, float],
                         goal_world: Tuple[float, float],
                         save_path: str = None,
                         show: bool = True):
    """
    Visualize Dijkstra vs A* comparison side by side.

    Args:
        grid_map: Occupancy grid map
        start_world: Start position (x, y)
        goal_world: Goal position (x, y)
        save_path: Optional path to save figure
        show: Whether to display the plot
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Run both planners
    path_d, stats_d = plan_path(grid_map, start_world, goal_world, PlannerType.DIJKSTRA)
    path_a, stats_a = plan_path(grid_map, start_world, goal_world, PlannerType.ASTAR)

    prob_grid = grid_map.grid

    # Occupancy grid background
    for ax in axes:
        im = ax.imshow(prob_grid, origin='lower', cmap='RdYlGn_r', vmin=0, vmax=1,
                       extent=[0, grid_map.width * grid_map.resolution,
                               0, grid_map.height * grid_map.resolution],
                       alpha=0.5)

    # Dijkstra
    ax = axes[0]
    if path_d:
        path_arr = np.array(path_d)
        ax.plot(path_arr[:, 0], path_arr[:, 1], 'b-', linewidth=3, label=f'Path ({len(path_d)} pts)')
    ax.scatter(*start_world, c='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(*goal_world, c='red', s=100, marker='*', label='Goal', zorder=5)
    ax.set_title(f'Dijkstra\nNodes: {stats_d.nodes_explored}, Len: {stats_d.path_length:.2f}m, Time: {stats_d.computation_time_ms:.1f}ms')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # A*
    ax = axes[1]
    if path_a:
        path_arr = np.array(path_a)
        ax.plot(path_arr[:, 0], path_arr[:, 1], 'r-', linewidth=3, label=f'Path ({len(path_a)} pts)')
    ax.scatter(*start_world, c='green', s=100, marker='o', zorder=5)
    ax.scatter(*goal_world, c='red', s=100, marker='*', zorder=5)
    ax.set_title(f'A* (Euclidean)\nNodes: {stats_a.nodes_explored}, Len: {stats_a.path_length:.2f}m, Time: {stats_a.computation_time_ms:.1f}ms')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Comparison bar chart
    ax = axes[2]
    metrics = ['Nodes Explored', 'Path Length (m)', 'Time (ms)']
    dijkstra_vals = [stats_d.nodes_explored, stats_d.path_length, stats_d.computation_time_ms]
    astar_vals = [stats_a.nodes_explored, stats_a.path_length, stats_a.computation_time_ms]

    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, dijkstra_vals, width, label='Dijkstra', color='skyblue')
    ax.bar(x + width/2, astar_vals, width, label='A*', color='lightcoral')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Value')
    ax.set_title('Planner Comparison')
    ax.legend()
    ax.grid(True, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

    return fig


if __name__ == "__main__":
    # Test planning
    from python_analysis.utils.config import EnvironmentConfig
    from python_analysis.utils.helpers import create_occupancy_grid

    config = EnvironmentConfig()
    grid_res = config.grid_resolution
    world_size = config.world_size
    grid_w = int(world_size[0] / grid_res)
    grid_h = int(world_size[1] / grid_res)

    # Create test grid with obstacles
    grid = create_occupancy_grid(grid_w, grid_h, 0.0)

    # Add some obstacles
    for r in range(grid_h//3, 2*grid_h//3):
        for c in range(grid_w//3, 2*grid_w//3):
            grid[r, c] = probability_to_log_odds(np.array([0.9]))[0]

    prob_grid = log_odds_to_probability(grid)
    occ_map = OccupancyGridMap(prob_grid, grid_res, (0, 0))

    start = (1.0, 1.0)
    goal = (9.0, 9.0)

    # Test Dijkstra
    path_d, stats_d = plan_path(occ_map, start, goal, PlannerType.DIJKSTRA)
    print(f"Dijkstra: found={stats_d.path_found}, nodes={stats_d.nodes_explored}, "
          f"length={stats_d.path_length:.2f}, time={stats_d.computation_time_ms:.2f}ms")

    # Test A*
    path_a, stats_a = plan_path(occ_map, start, goal, PlannerType.ASTAR)
    print(f"A*: found={stats_a.path_found}, nodes={stats_a.nodes_explored}, "
          f"length={stats_a.path_length:.2f}, time={stats_a.computation_time_ms:.2f}ms")

    # Compare
    comparison = compare_planners(occ_map, start, goal, runs=3)
    print("\nComparison (avg over 3 runs):")
    for planner, stats in comparison.items():
        print(f"  {planner}: nodes={stats['avg_nodes_explored']:.0f}, "
              f"len={stats['avg_path_length']:.2f}, time={stats['avg_computation_time_ms']:.2f}ms, "
              f"success={stats['success_rate']:.0%}")