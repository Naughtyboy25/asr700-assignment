"""
Tests for Component B: consensus, swarm intelligence (PSO/ACO), and the
ML navigation policy.

Covers the offline reproducible simulation
(python_simulations/component_b/run_component_b.py) and the Webots swarm
controller modules (webots_project/controllers/swarm_agents/).
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


sim_b = _load_module("run_component_b", "python_simulations/component_b/run_component_b.py")

from consensus import (  # noqa: E402
    AgentState,
    ConsensusProtocol,
    ConsensusType,
    LeaderElection,
    simulate_consensus_dynamics,
)
from ml_policy import (  # noqa: E402
    NavigationCNN,
    compute_classification_metrics,
    compute_confusion_matrix,
    discretize_actions,
    generate_training_data,
)
from swarm_intelligence import CoverageACO, ParticleSwarmOptimization  # noqa: E402

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Offline simulation: consensus
# ---------------------------------------------------------------------------

class TestOfflineConsensus:
    def test_variance_decreases(self):
        _, hist = sim_b.run_consensus()
        var_early = hist[:10, :, 0].var(axis=1).mean()
        var_late = hist[-10:, :, 0].var(axis=1).mean()
        assert var_late < var_early * 0.1

    def test_agents_stay_in_world(self):
        agents, _ = sim_b.run_consensus()
        for a in agents:
            assert 0 <= a.x <= sim_b.WORLD
            assert 0 <= a.y <= sim_b.WORLD


# ---------------------------------------------------------------------------
# Offline simulation: PSO coverage
# ---------------------------------------------------------------------------

class TestOfflinePSO:
    def test_coverage_progress_increases(self):
        _, _, _, cov_hist = sim_b.run_pso()
        assert cov_hist[-1] >= cov_hist[0]
        assert cov_hist[-1] > 0.9  # demo reaches near-full coverage


# ---------------------------------------------------------------------------
# Offline simulation: neural navigation policy
# ---------------------------------------------------------------------------

class TestOfflineNavNet:
    def test_dataset_shapes_and_labels(self):
        X, y = sim_b.make_dataset(200)
        assert X.shape == (200, 6)
        assert y.shape == (200,)
        assert set(np.unique(y)).issubset({0, 1, 2, 3, 4})

    def test_forward_output_shape(self):
        model = sim_b.NavNet()
        out = model(torch.randn(7, 6))
        assert out.shape == (7, 5)

    def test_training_improves_accuracy(self):
        np.random.seed(0)
        torch.manual_seed(0)
        _, _, acc, cm = sim_b.train_nn(epochs=3)
        assert 0.5 < acc <= 1.0
        assert cm.shape == (5, 5)
        assert cm.sum() > 0


# ---------------------------------------------------------------------------
# Webots swarm modules (pure logic, no Webots runtime needed)
# ---------------------------------------------------------------------------

class TestConsensusProtocol:
    def test_average_consensus_converges(self):
        protocol = ConsensusProtocol()
        agents = [
            AgentState(0, (0, 0), value=1.0),
            AgentState(1, (1, 0), value=2.0),
            AgentState(2, (0, 1), value=3.0),
            AgentState(3, (1, 1), value=4.0),
        ]
        final_agents, info = protocol.run_consensus(
            agents, ConsensusType.AVERAGE, max_iterations=500, tolerance=1e-4
        )
        assert info["converged"]
        # AVERAGE consensus updates the vector value; all agents must agree
        # on the mean of the initial vectors (preserved by the iteration)
        vectors = np.array([a.vector_value for a in final_agents])
        assert vectors.std(axis=0).max() < 1e-3
        expected = np.mean([a.vector_value for a in agents], axis=0)
        assert np.allclose(vectors.mean(axis=0), expected, atol=1e-2)

    def test_simulate_dynamics_complete_topology(self):
        np.random.seed(42)
        result = simulate_consensus_dynamics(num_agents=5, topology="complete",
                                             alpha=0.1, max_iter=500)
        assert result["convergence_time"] < 500
        assert result["spectral_gap"] > -1e-9  # allow float noise around zero

    def test_bully_election_picks_highest_id(self):
        agents = [AgentState(i, (i, 0)) for i in range(4)]
        leader = LeaderElection.bully_election(agents)
        assert leader == 3


class TestSwarmIntelligence:
    def test_pso_minimizes_sphere(self):
        np.random.seed(42)
        pso = ParticleSwarmOptimization(bounds=(-5 * np.ones(2), 5 * np.ones(2)))
        pso.max_iter = 60
        result = pso.optimize(lambda x: float(np.sum(x ** 2)), dim=2, num_particles=20)
        assert result.best_fitness < 0.5
        assert np.linalg.norm(result.best_position) < 1.0

    def test_coverage_aco_runs_and_reports(self):
        np.random.seed(42)
        grid = np.zeros((20, 20), dtype=np.uint8)
        grid[8:12, 8:12] = 1  # obstacle
        aco = CoverageACO(grid, num_agents=2, start_positions=[(1, 1), (18, 18)],
                          num_ants=5, max_iter=5)
        result = aco.optimize()
        assert result["best_coverage"] > 0
        assert 0 < result["coverage_ratio"] <= 1.0
        assert len(result["convergence"]) == 5


class TestMLPolicy:
    def test_training_data_shapes(self):
        lidar, camera, targets = generate_training_data(50, None)
        assert lidar.shape == (50, 360)
        assert camera.shape == (50, 64, 64, 3)  # HWC
        assert targets.shape == (50, 2)

    def test_cnn_forward_both_camera_layouts(self):
        model = NavigationCNN()
        model.eval()
        lidar = torch.randn(4, 360)
        with torch.no_grad():
            out_nchw = model(lidar, torch.randn(4, 3, 64, 64))
            out_nhwc = model(lidar, torch.randn(4, 64, 64, 3))
        assert out_nchw.shape == (4, 2)
        assert out_nhwc.shape == (4, 2)

    def test_discretize_actions_classes(self):
        actions = np.array([
            [0.5, 0.0],    # 0: forward
            [0.5, 1.0],    # 1: left
            [0.5, -1.0],   # 2: right
            [0.0, 0.0],    # 3: stop
            [-0.5, 0.0],   # 4: backward
        ])
        labels = discretize_actions(actions)
        assert labels.tolist() == [0, 1, 2, 3, 4]

    def test_confusion_matrix_and_metrics(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])
        cm = compute_confusion_matrix(y_true, y_pred, num_classes=3)
        assert cm.shape == (3, 3)
        assert cm.sum() == 6
        metrics = compute_classification_metrics(cm)
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert abs(metrics["accuracy"] - 4 / 6) < 1e-9
