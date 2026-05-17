"""Community Detection Module - Leiden Algorithm Implementation.

This module provides a robust implementation of the Leiden algorithm for community
detection. It improves upon Louvain by introducing a bounded Refinement Phase,
guaranteeing well-connected communities without the need for post-hoc splitting.
"""

from typing import List, Optional, Union, Dict
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

    def _optimize_partition(self, adj_list: Dict[int, Dict[int, float]], labels: np.ndarray,
                            node_degrees: np.ndarray, comm_degrees: np.ndarray,
                            m: float, bounds: Optional[np.ndarray] = None,
                            max_passes: int = 10) -> None:
        """Locally move nodes to maximize modularity.

        If `bounds` is provided (Refinement Phase), nodes are restricted from
        merging with communities outside of their overarching bounded community.
        """
        n = len(labels)
        nodes_in_comm = {}
        for node, comm in enumerate(labels):
            if comm not in nodes_in_comm:
                nodes_in_comm[comm] = set()
            nodes_in_comm[comm].add(node)

        for _ in range(max_passes):
            improved = False
            order = np.random.permutation(n)

            for node in order:
                current_comm = labels[node]
                k_i = node_degrees[node]

                # Find connected neighbors and their communities
                node_neighbors = adj_list.get(node, {})
                if not node_neighbors:
                    continue

                # Identify candidate target communities and their weights to current node
                comm_weights = {}
                for neighbor, weight in node_neighbors.items():
                    c = labels[neighbor]
                    comm_weights[c] = comm_weights.get(c, 0.0) + weight

                candidates = set(comm_weights.keys())

                # REFINEMENT RESTRICTION: Only merge within the same overarching bound
                if bounds is not None:
                    node_bound = bounds[node]
                    valid_candidates = set()
                    for c in candidates:
                        # Check the bound of a representative node in the target community
                        # We use the first node in nodes_in_comm[c] as representative
                        rep_node = next(iter(nodes_in_comm[c]))
                        if bounds[rep_node] == node_bound:
                            valid_candidates.add(c)
                    candidates = valid_candidates

                best_comm = current_comm
                best_delta = 0.0

                k_i_in_curr = comm_weights.get(current_comm, 0.0)

                for target_comm in candidates:
                    if target_comm == current_comm:
                        continue

                    k_i_in_target = comm_weights.get(target_comm, 0.0)

                    # Exact modularity difference for moving node to target_comm
                    delta = (k_i_in_target - k_i_in_curr) / m - self.resolution * k_i * (comm_degrees[target_comm] - comm_degrees[current_comm] + k_i) / (2 * m ** 2)

                    if delta > best_delta:
                        best_delta = delta
                        best_comm = target_comm

                # If improvement is significant, apply the move
                if best_delta > 1e-8:
                    labels[node] = best_comm
                    nodes_in_comm[current_comm].remove(node)
                    if not nodes_in_comm[current_comm]:
                        del nodes_in_comm[current_comm]
                    if best_comm not in nodes_in_comm:
                        nodes_in_comm[best_comm] = set()
                    nodes_in_comm[best_comm].add(node)

                    comm_degrees[current_comm] -= k_i
                    comm_degrees[best_comm] += k_i
                    improved = True

            if not improved:
                break

    def _aggregate_graph(self, adj_list: Dict[int, Dict[int, float]], labels: np.ndarray) -> Dict[int, Dict[int, float]]:
        """Create a new, smaller adjacency list where nodes are the discovered communities."""
        new_adj = {}
        for node, neighbors in adj_list.items():
            comm_i = labels[node]
            if comm_i not in new_adj:
                new_adj[comm_i] = {}
            for neighbor, weight in neighbors.items():
                comm_j = labels[neighbor]
                new_adj[comm_i][comm_j] = new_adj[comm_i].get(comm_j, 0.0) + weight
        return new_adj

    def fit(self, adjacency: Union[np.ndarray, List[List[float]], Dict[int, Dict[int, float]]]) -> np.ndarray:
        """Fit the Leiden algorithm to an adjacency representation.

        Args:
            adjacency: Adjacency matrix (n x n) or adjacency list.

        Returns:
            1D NumPy array of community labels for each original node (0-indexed).
        """
        if isinstance(adjacency, (np.ndarray, list)):
            # Convert dense to sparse adjacency list for consistency
            adj_list = {}
            adjacency_np = np.array(adjacency, dtype=np.float64)
            n_original = adjacency_np.shape[0]
            for i in range(n_original):
                adj_list[i] = {}
                for j in np.where(adjacency_np[i] > 0)[0]:
                    adj_list[i][j] = adjacency_np[i, j]
        else:
            adj_list = adjacency
            n_original = max(adj_list.keys()) + 1 if adj_list else 0

        if n_original <= 1:
            return np.zeros(n_original, dtype=int)

        # Track the community assignment for the ORIGINAL dataset
        global_labels = np.arange(n_original)
        current_adj = adj_list

        for _ in range(self.max_iter):
            n_current = len(set(global_labels)) if current_adj else 0
            # Map current nodes (communities) to 0..n_current-1
            unique_nodes = sorted(current_adj.keys())
            if not unique_nodes:
                break

            node_map = {node: i for i, node in enumerate(unique_nodes)}
            n_current = len(unique_nodes)

            # Normalize current_adj keys to 0..n_current-1
            norm_adj = {}
            for u, neighbors in current_adj.items():
                u_idx = node_map[u]
                norm_adj[u_idx] = {node_map[v]: w for v, w in neighbors.items() if v in node_map}

            m = sum(sum(neigh.values()) for neigh in norm_adj.values()) / 2.0

            # Stop if there are no edges left to evaluate
            if m <= 0:
                break

            node_degrees = np.array([sum(neigh.values()) for idx, neigh in sorted(norm_adj.items())])

            # PHASE 1: Fast Local Move (Louvain step)
            partition = np.arange(n_current)
            comm_degrees = node_degrees.copy()
            self._optimize_partition(norm_adj, partition, node_degrees, comm_degrees, m)

            # PHASE 2: Refinement Phase (Leiden strictly)
            refined_partition = np.arange(n_current)
            ref_comm_degrees = node_degrees.copy()
            # Pass 'partition' as bounds: nodes can only merge if they share the same upper partition
            self._optimize_partition(
                norm_adj, refined_partition, node_degrees, ref_comm_degrees,
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
            # First map global_labels to current norm_adj indices
            current_mapping = np.array([node_map[l] for l in global_labels])
            global_labels = contiguous_refined[current_mapping]

            # PHASE 3: Network Aggregation
            current_adj = self._aggregate_graph(norm_adj, contiguous_refined)

        return global_labels


def leiden(adjacency: Union[np.ndarray, List[List[float]], Dict[int, Dict[int, float]]], resolution: float = 1.0,
           max_iter: int = 10, random_state: Optional[int] = None) -> np.ndarray:
    """Wrapper function for Leiden community detection.

    Args:
        adjacency: Adjacency representation.
        resolution: Resolution parameter (higher values = more communities).
        max_iter: Maximum number of macroscopic aggregation iterations.
        random_state: Random seed for reproducibility.

    Returns:
        Community labels for each node.
    """
    algorithm = LeidenAlgorithm(resolution=resolution, max_iter=max_iter, random_state=random_state)
    return algorithm.fit(adjacency)
