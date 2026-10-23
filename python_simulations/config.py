"""
Configuration constants for ASR700 Assignment.
Centralized configuration for all components.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class RobotConfig:
    """Differential drive robot physical parameters."""
    wheel_radius: float = 0.02  # meters
    wheel_base: float = 0.052  # meters (distance between wheels)
    max_linear_velocity: float = 0.12  # m/s
    max_angular_velocity: float = 2.0  # rad/s
    encoder_resolution: int = 1000  # ticks per revolution


@dataclass(frozen=True)
class SensorConfig:
    """Sensor configuration parameters."""
    # LiDAR
    lidar_fov: float = 270.0  # degrees
    lidar_resolution: float = 0.5  # degrees
    lidar_max_range: float = 5.0  # meters
    lidar_min_range: float = 0.02  # meters

    # Camera
    camera_width: int = 640
    camera_height: int = 480
    camera_fov: float = 60.0  # degrees


@dataclass(frozen=True)
class EnvironmentConfig:
    """Simulation environment parameters."""
    world_size: Tuple[float, float] = (10.0, 10.0)  # meters
    grid_resolution: float = 0.05  # meters per cell
    obstacle_inflation_radius: float = 0.15  # meters
    start_position: Tuple[float, float] = (1.0, 1.0)
    goal_position: Tuple[float, float] = (9.0, 9.0)


@dataclass(frozen=True)
class PIDConfig:
    """PID controller default gains."""
    # Linear velocity PID
    kp_linear: float = 2.0
    ki_linear: float = 0.1
    kd_linear: float = 0.5

    # Angular velocity PID
    kp_angular: float = 3.0
    ki_angular: float = 0.05
    kd_angular: float = 0.8


@dataclass(frozen=True)
class PathPlanningConfig:
    """Path planning algorithm parameters."""
    heuristic_weight: float = 1.0  # A* heuristic weight
    diagonal_movement: bool = True
    replan_threshold: float = 0.5  # meters - trigger replanning if tracking error exceeds this


@dataclass(frozen=True)
class SwarmConfig:
    """Multi-agent swarm parameters."""
    num_agents: int = 4
    communication_range: float = 2.0  # meters
    consensus_alpha: float = 0.1  # Consensus convergence rate
    pso_inertia: float = 0.7
    pso_cognitive: float = 1.5
    pso_social: float = 1.5
    pso_max_iterations: int = 100


@dataclass(frozen=True)
class MLConfig:
    """Machine learning configuration."""
    input_channels: int = 3  # LiDAR (1) + Camera RGB (3) = 4, but we'll use processed
    lidar_input_dim: int = 360  # LiDAR scans per timestep
    camera_input_shape: Tuple[int, int, int] = (3, 64, 64)  # Downsampled camera (C, H, W)
    hidden_dim: int = 128
    output_dim: int = 2  # Linear velocity, Angular velocity
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 50
    train_split: float = 0.8
    random_seed: int = 42


# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
WEBOTS_PROJECT = PROJECT_ROOT / "webots_project"
PYTHON_ANALYSIS = PROJECT_ROOT / "python_analysis"
DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "report"

WORLDS_DIR = WEBOTS_PROJECT / "worlds"
CONTROLLERS_DIR = WEBOTS_PROJECT / "controllers"
RESOURCES_DIR = WEBOTS_PROJECT / "resources"
MAPS_DIR = RESOURCES_DIR / "maps"
MODELS_DIR = RESOURCES_DIR / "models"

SENSOR_LOGS_DIR = DATA_DIR / "sensor_logs"
TRAJECTORIES_DIR = DATA_DIR / "trajectories"
ML_DATASETS_DIR = DATA_DIR / "ml_datasets"

FIGURES_DIR = REPORT_DIR / "figures"

# Default configurations (singletons)
ROBOT_CONFIG = RobotConfig()
SENSOR_CONFIG = SensorConfig()
ENV_CONFIG = EnvironmentConfig()
PID_CONFIG = PIDConfig()
PLANNING_CONFIG = PathPlanningConfig()
SWARM_CONFIG = SwarmConfig()
ML_CONFIG = MLConfig()


def ensure_directories():
    """Create all necessary directories."""
    dirs = [
        WORLDS_DIR, CONTROLLERS_DIR, RESOURCES_DIR, MAPS_DIR, MODELS_DIR,
        SENSOR_LOGS_DIR, TRAJECTORIES_DIR, ML_DATASETS_DIR, FIGURES_DIR,
        PYTHON_ANALYSIS / "component_a_analysis",
        PYTHON_ANALYSIS / "component_b_analysis",
        PYTHON_ANALYSIS / "training",
        PYTHON_ANALYSIS / "evaluation",
        PYTHON_ANALYSIS / "utils",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    print("All directories created successfully.")
    print(f"Project root: {PROJECT_ROOT}")