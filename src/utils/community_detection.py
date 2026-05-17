"""Community Detection Module - Leiden Algorithm Implementation.

This module provides a robust implementation of the Leiden algorithm for community
detection. It improves upon Louvain by introducing a bounded Refinement Phase,
guaranteeing well-connected communities without the need for post-hoc splitting.
"""

from typing import List, Optional, Union
import numpy as np


class LeidenAlgorithm:
    """Leiden Algorithm for Community Detection."""

    def __init__(self, resolution: float = 1.0, max_iter: int = 10, random_state: Optional[int] = None):
        """Initialize the Leiden algorithm.

        Args:
            resolution: Resolution parameter (higher values lead to more, smaller communities).
            max_iter: Maximum number of aggregation passes.
            random_state: Random seed for reproducibility.
        """
        self.resolution = resolution
        self.max_iter = max_iter
        if random_state is not None:
            np.random.seed(random_state)

    def _optimize_partition(self, adjacency: np.ndarray, labels: np.ndarray,
                            node_degrees: np.ndarray, comm_degrees: np.ndarray,
                            m: float, bounds: Optional[np.ndarray] = None,
                            max_passes: int = 10) -> None:
        """Locally move nodes to maximize modularity.

        If `bounds` is provided (Refinement Phase), nodes are restricted from
        merging with communities outside of their overarching bounded community.
        """
        n = adjacency.shape[0]

        for _ in range(max_passes):
            improved = False
            order = np.random.permutation(n)

            for node in order:
                current_comm = labels[node]
                k_i = node_degrees[node]

                # Find connected neighbors
                neighbors = np.where(adjacency[node] > 0)[0]
                if len(neighbors) == 0:
                    continue

                # Identify candidate target communities
                candidates = set(labels[neighbors])

                # REFINEMENT RESTRICTION: Only merge within the same overarching bound
                if bounds is not None:
                    node_bound = bounds[node]
                    valid_candidates = set()
                    for c in candidates:
                        # Check the bound of a representative node in the target community
                        rep_node = np.where(labels == c)[0][0]
                        if bounds[rep_node] == node_bound:
                            valid_candidates.add(c)
                    candidates = valid_candidates

                if current_comm in candidates:
                    candidates.remove(current_comm)

                best_comm = current_comm
                best_delta = 0.0

                k_i_in_curr = np.sum(adjacency[node, labels == current_comm])

                for target_comm in candidates:
                    k_i_in_target = np.sum(adjacency[node, labels == target_comm])

                    # Exact modularity difference for moving node to target_comm
                    delta = (k_i_in_target - k_i_in_curr) / m - self.resolution * k_i * (comm_degrees[target_comm] - comm_degrees[current_comm] + k_i) / (2 * m ** 2)

                    if delta > best_delta:
                        best_delta = delta
                        best_comm = target_comm

                # If improvement is significant, apply the move
                if best_delta > 1e-8:
                    labels[node] = best_comm
                    comm_degrees[current_comm] -= k_i
                    comm_degrees[best_comm] += k_i
                    improved = True

            if not improved:
                break

    def _aggregate_graph(self, adjacency: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Create a new, smaller adjacency matrix where nodes are the discovered communities."""
        n_communities = np.max(labels) + 1
        aggregated = np.zeros((n_communities, n_communities), dtype=np.float64)

        for i in range(n_communities):
            nodes_i = np.where(labels == i)[0]
            for j in range(i, n_communities):
                nodes_j = np.where(labels == j)[0]

                # Sum weights between communities (or within, if i == j)
                weight = np.sum(adjacency[np.ix_(nodes_i, nodes_j)])
                aggregated[i, j] = weight
                aggregated[j, i] = weight

        return aggregated

    def fit(self, adjacency: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        """Fit the Leiden algorithm to an adjacency matrix.

        Args:
            adjacency: Adjacency matrix (n x n) representing edge weights.

        Returns:
            1D NumPy array of community labels for each original node (0-indexed).
        """
        adjacency = np.array(adjacency, dtype=np.float64)
        n_original = adjacency.shape[0]

        if n_original <= 1:
            return np.zeros(n_original, dtype=int)

        # Track the community assignment for the ORIGINAL dataset
        global_labels = np.arange(n_original)
        current_adj = adjacency.copy()

        for _ in range(self.max_iter):
            n_current = current_adj.shape[0]
            m = np.sum(current_adj) / 2.0

            # Stop if there are no edges left to evaluate
            if m <= 0:
                break

            node_degrees = np.sum(current_adj, axis=1)

            # PHASE 1: Fast Local Move (Louvain step)
            partition = np.arange(n_current)
            comm_degrees = node_degrees.copy()
            self._optimize_partition(current_adj, partition, node_degrees, comm_degrees, m)

            # PHASE 2: Refinement Phase (Leiden strictly)
            refined_partition = np.arange(n_current)
            ref_comm_degrees = node_degrees.copy()
            # Pass 'partition' as bounds: nodes can only merge if they share the same upper partition
            self._optimize_partition(
                current_adj, refined_partition, node_degrees, ref_comm_degrees,
                m, bounds=partition, max_passes=1
            )

            # Remap refined labels to contiguous 0-indexed integers
            unique_labels = np.unique(refined_partition)

            # Stop if no nodes merged (convergence reached)
            if len(unique_labels) == n_current:
                break

            label_mapping = {old: new for new, old in enumerate(unique_labels)}
            contiguous_refined = np.array([label_mapping[l] for l in refined_partition])

            # Update the global tracker mapping previous components to new super-nodes
            global_labels = contiguous_refined[global_labels]

            # PHASE 3: Network Aggregation
            current_adj = self._aggregate_graph(current_adj, contiguous_refined)

        return global_labels


def leiden(adjacency: Union[np.ndarray, List[List[float]]], resolution: float = 1.0,
           max_iter: int = 10, random_state: Optional[int] = None) -> np.ndarray:
    """Wrapper function for Leiden community detection.

    Args:
        adjacency: Adjacency matrix (n x n).
        resolution: Resolution parameter (higher values = more communities).
        max_iter: Maximum number of macroscopic aggregation iterations.
        random_state: Random seed for reproducibility.

    Returns:
        Community labels for each node.
    """
    algorithm = LeidenAlgorithm(resolution=resolution, max_iter=max_iter, random_state=random_state)
    return algorithm.fit(adjacency)
