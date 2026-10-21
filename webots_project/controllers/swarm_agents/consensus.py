"""
Consensus Algorithms for ASR700 Component B.

Implements average consensus and leader election for multi-agent coordination.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Callable
from enum import Enum
from python_analysis.utils.config import SwarmConfig


class ConsensusType(Enum):
    """Consensus algorithm types."""
    AVERAGE = "average"
    MAX = "max"
    MIN = "min"
    LEADER_ELECTION = "leader_election"


@dataclass
class AgentState:
    """State of a single agent in the swarm."""
    agent_id: int
    position: Tuple[float, float]
    velocity: Tuple[float, float] = (0.0, 0.0)
    heading: float = 0.0
    value: float = 0.0  # Scalar value for consensus
    vector_value: np.ndarray = None  # Vector value for consensus

    def __post_init__(self):
        if self.vector_value is None:
            self.vector_value = np.array(self.position)


class ConsensusProtocol:
    """
    Distributed consensus protocol for multi-agent systems.

    Implements:
    - Average consensus: x_i(k+1) = x_i(k) + alpha * sum(x_j - x_i) for j in neighbors
    - Max/Min consensus
    - Leader election
    """

    def __init__(self, config: SwarmConfig = None):
        self.config = config or SwarmConfig()
        self.alpha = self.config.consensus_alpha  # Convergence rate
        self.communication_range = self.config.communication_range

    def get_neighbors(self, agents: List[AgentState],
                      agent_idx: int) -> List[int]:
        """
        Get neighbors within communication range.

        Args:
            agents: List of all agent states
            agent_idx: Index of current agent

        Returns:
            List of neighbor indices
        """
        neighbors = []
        pos_i = np.array(agents[agent_idx].position)

        for j, agent in enumerate(agents):
            if j == agent_idx:
                continue
            pos_j = np.array(agent.position)
            dist = np.linalg.norm(pos_j - pos_i)
            if dist <= self.communication_range:
                neighbors.append(j)

        return neighbors

    def average_consensus_step(self, agents: List[AgentState],
                               agent_idx: int,
                               dt: float = 1.0) -> float:
        """
        Single step of average consensus for scalar value.

        Args:
            agents: All agent states
            agent_idx: Current agent index
            dt: Time step

        Returns:
            Updated value for agent
        """
        neighbors = self.get_neighbors(agents, agent_idx)

        if not neighbors:
            return agents[agent_idx].value

        current = agents[agent_idx].value
        neighbor_values = [agents[j].value for j in neighbors]

        # Consensus update: x_i = x_i + alpha * sum(x_j - x_i)
        consensus_sum = sum(v - current for v in neighbor_values)
        new_value = current + self.alpha * consensus_sum * dt

        return new_value

    def average_consensus_vector_step(self, agents: List[AgentState],
                                      agent_idx: int,
                                      dt: float = 1.0) -> np.ndarray:
        """
        Single step of average consensus for vector value (position).

        Args:
            agents: All agent states
            agent_idx: Current agent index
            dt: Time step

        Returns:
            Updated vector value
        """
        neighbors = self.get_neighbors(agents, agent_idx)

        if not neighbors:
            return agents[agent_idx].vector_value

        current = agents[agent_idx].vector_value
        consensus_sum = np.zeros_like(current)

        for j in neighbors:
            consensus_sum += agents[j].vector_value - current

        new_value = current + self.alpha * consensus_sum * dt
        return new_value

    def max_consensus_step(self, agents: List[AgentState],
                           agent_idx: int) -> float:
        """Max consensus step."""
        neighbors = self.get_neighbors(agents, agent_idx)
        if not neighbors:
            return agents[agent_idx].value

        current = agents[agent_idx].value
        neighbor_values = [agents[j].value for j in neighbors]
        return max([current] + neighbor_values)

    def min_consensus_step(self, agents: List[AgentState],
                           agent_idx: int) -> float:
        """Min consensus step."""
        neighbors = self.get_neighbors(agents, agent_idx)
        if not neighbors:
            return agents[agent_idx].value

        current = agents[agent_idx].value
        neighbor_values = [agents[j].value for j in neighbors]
        return min([current] + neighbor_values)

    def run_consensus(self, agents: List[AgentState],
                      consensus_type: ConsensusType = ConsensusType.AVERAGE,
                      max_iterations: int = 1000,
                      tolerance: float = 1e-4,
                      dt: float = 1.0) -> Tuple[List[AgentState], Dict]:
        """
        Run consensus algorithm to convergence.

        Args:
            agents: Initial agent states
            consensus_type: Type of consensus
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            dt: Time step

        Returns:
            (final_agents, convergence_info)
        """
        agents = [AgentState(a.agent_id, a.position, a.velocity, a.heading,
                             a.value, a.vector_value.copy() if a.vector_value is not None else None)
                  for a in agents]

        convergence_history = []

        for iteration in range(max_iterations):
            max_change = 0.0

            # Store old values
            old_values = [a.value for a in agents]
            old_vectors = [a.vector_value.copy() for a in agents]

            # Update each agent
            new_agents = []
            for i, agent in enumerate(agents):
                new_agent = AgentState(
                    agent.agent_id, agent.position, agent.velocity, agent.heading,
                    agent.value, agent.vector_value.copy()
                )

                if consensus_type == ConsensusType.AVERAGE:
                    if agent.vector_value is not None:
                        new_agent.vector_value = self.average_consensus_vector_step(agents, i, dt)
                    else:
                        new_agent.value = self.average_consensus_step(agents, i, dt)
                elif consensus_type == ConsensusType.MAX:
                    new_agent.value = self.max_consensus_step(agents, i)
                elif consensus_type == ConsensusType.MIN:
                    new_agent.value = self.min_consensus_step(agents, i)

                new_agents.append(new_agent)

            agents = new_agents

            # Check convergence
            if consensus_type == ConsensusType.AVERAGE:
                if agents[0].vector_value is not None:
                    changes = [np.linalg.norm(agents[i].vector_value - old_vectors[i])
                               for i in range(len(agents))]
                else:
                    changes = [abs(agents[i].value - old_values[i]) for i in range(len(agents))]
            else:
                changes = [abs(agents[i].value - old_values[i]) for i in range(len(agents))]

            max_change = max(changes)
            convergence_history.append(max_change)

            if max_change < tolerance:
                break

        # Compute final consensus value
        if consensus_type == ConsensusType.AVERAGE:
            if agents[0].vector_value is not None:
                final_consensus = np.mean([a.vector_value for a in agents], axis=0)
            else:
                final_consensus = np.mean([a.value for a in agents])
        elif consensus_type == ConsensusType.MAX:
            final_consensus = max(a.value for a in agents)
        elif consensus_type == ConsensusType.MIN:
            final_consensus = min(a.value for a in agents)

        info = {
            'iterations': iteration + 1,
            'converged': max_change < tolerance,
            'final_consensus': final_consensus,
            'convergence_history': convergence_history,
            'max_change': max_change
        }

        return agents, info


class LeaderElection:
    """
    Leader election protocols for swarm coordination.

    Implements:
    - Bully algorithm (highest ID wins)
    - Randomized leader election
    - Distance-based (closest to target)
    """

    @staticmethod
    def bully_election(agents: List[AgentState]) -> int:
        """Bully algorithm: highest agent ID becomes leader."""
        return max(agents, key=lambda a: a.agent_id).agent_id

    @staticmethod
    def random_election(agents: List[AgentState], seed: int = None) -> int:
        """Random leader election."""
        if seed is not None:
            np.random.seed(seed)
        return np.random.choice([a.agent_id for a in agents])

    @staticmethod
    def closest_to_target_election(agents: List[AgentState],
                                   target: Tuple[float, float]) -> int:
        """Elect agent closest to target position."""
        target_arr = np.array(target)
        return min(agents, key=lambda a: np.linalg.norm(np.array(a.position) - target_arr)).agent_id

    @staticmethod
    def highest_connectivity_election(agents: List[AgentState],
                                      comm_range: float) -> int:
        """Elect agent with most neighbors."""
        max_neighbors = -1
        leader_id = agents[0].agent_id

        for i, agent in enumerate(agents):
            count = 0
            for j, other in enumerate(agents):
                if i != j:
                    dist = np.linalg.norm(np.array(agent.position) - np.array(other.position))
                    if dist <= comm_range:
                        count += 1
            if count > max_neighbors:
                max_neighbors = count
                leader_id = agent.agent_id

        return leader_id


def simulate_consensus_dynamics(num_agents: int = 5,
                                topology: str = "complete",
                                alpha: float = 0.1,
                                max_iter: int = 200) -> Dict:
    """
    Simulate consensus dynamics for analysis.

    Args:
        num_agents: Number of agents
        topology: "complete", "ring", "line", "random"
        alpha: Consensus gain
        max_iter: Maximum iterations

    Returns:
        Dictionary with trajectories and convergence metrics
    """
    # Initialize agents with random values
    agents = []
    for i in range(num_agents):
        agents.append(AgentState(
            agent_id=i,
            position=(np.random.rand() * 10, np.random.rand() * 10),
            value=np.random.rand() * 10
        ))

    # Build adjacency matrix based on topology
    if topology == "complete":
        adj = np.ones((num_agents, num_agents)) - np.eye(num_agents)
    elif topology == "ring":
        adj = np.zeros((num_agents, num_agents))
        for i in range(num_agents):
            adj[i, (i+1)%num_agents] = 1
            adj[i, (i-1)%num_agents] = 1
    elif topology == "line":
        adj = np.zeros((num_agents, num_agents))
        for i in range(num_agents-1):
            adj[i, i+1] = 1
            adj[i+1, i] = 1
    elif topology == "random":
        adj = (np.random.rand(num_agents, num_agents) < 0.3).astype(float)
        adj = (adj + adj.T) / 2
        np.fill_diagonal(adj, 0)
    else:
        raise ValueError(f"Unknown topology: {topology}")

    # Degree matrix
    degree = np.diag(np.sum(adj, axis=1))

    # Laplacian
    L = degree - adj

    # Consensus dynamics: x(k+1) = (I - alpha*L) x(k)
    A = np.eye(num_agents) - alpha * L

    # Simulate
    values = np.array([a.value for a in agents])
    trajectories = [values.copy()]

    for _ in range(max_iter):
        values = A @ values
        trajectories.append(values.copy())

        # Check convergence
        if np.max(values) - np.min(values) < 1e-6:
            break

    trajectories = np.array(trajectories)

    return {
        'trajectories': trajectories,
        'convergence_time': len(trajectories),
        'final_values': values,
        'consensus_value': np.mean(values),
        'eigenvalues': np.linalg.eigvals(A),
        'spectral_gap': 1 - np.max(np.abs(np.linalg.eigvals(A)[1:]))
    }


if __name__ == "__main__":
    # Test consensus
    config = SwarmConfig()
    protocol = ConsensusProtocol(config)

    # Create test agents
    agents = [
        AgentState(0, (0, 0), value=1.0),
        AgentState(1, (1, 0), value=2.0),
        AgentState(2, (0, 1), value=3.0),
        AgentState(3, (1, 1), value=4.0),
    ]

    # Run average consensus
    final_agents, info = protocol.run_consensus(
        agents, ConsensusType.AVERAGE, max_iterations=200, tolerance=1e-4
    )

    print(f"Consensus converged: {info['converged']}")
    print(f"Iterations: {info['iterations']}")
    print(f"Final consensus value: {info['final_consensus']:.4f}")
    print(f"Individual values: {[a.value for a in final_agents]}")

    # Test leader election
    leader = LeaderElection.bully_election(agents)
    print(f"Bully leader: Agent {leader}")

    leader = LeaderElection.closest_to_target_election(agents, (5, 5))
    print(f"Closest to target leader: Agent {leader}")

    # Test dynamics simulation
    sim = simulate_consensus_dynamics(num_agents=6, topology="ring", alpha=0.2)
    print(f"\nDynamics simulation: converged in {sim['convergence_time']} steps")
    print(f"Consensus value: {sim['consensus_value']:.4f}")
    print(f"Spectral gap: {sim['spectral_gap']:.4f}")