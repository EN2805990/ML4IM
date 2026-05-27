from __future__ import annotations

from collections import deque
from typing import Iterable, Set

import networkx as nx


def covered_nodes(
    graph: nx.Graph,
    seeds: Iterable[int],
    cover_radius: int = 1,
) -> Set[int]:
    """Return nodes covered by seed nodes within ``cover_radius`` hops."""
    if cover_radius < 0:
        raise ValueError("cover_radius must be non-negative")

    covered: Set[int] = set()
    for seed in seeds:
        if seed not in graph:
            continue

        queue = deque([(seed, 0)])
        seen = {seed}
        while queue:
            node, depth = queue.popleft()
            covered.add(node)
            if depth == cover_radius:
                continue
            for nbr in graph.neighbors(node):
                if nbr not in seen:
                    seen.add(nbr)
                    queue.append((nbr, depth + 1))

    return covered


def greedy_max_cover(
    graph: nx.Graph,
    k: int,
    cover_radius: int = 1,
) -> tuple[list[int], int]:
    """Greedy approximation for graph maximum coverage.

    The candidate set is all nodes in ``graph``. At each step, the node with the
    largest marginal gain is selected.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0 or graph.number_of_nodes() == 0:
        return [], 0

    selected: list[int] = []
    covered: set[int] = set()
    candidates = set(graph.nodes())

    for _ in range(min(k, graph.number_of_nodes())):
        best_node = None
        best_gain = -1
        best_new_covered: set[int] = set()

        for node in candidates:
            node_covered = covered_nodes(graph, [node], cover_radius)
            new_covered = node_covered - covered
            gain = len(new_covered)
            if gain > best_gain:
                best_node = node
                best_gain = gain
                best_new_covered = new_covered

        if best_node is None or best_gain <= 0:
            break

        selected.append(best_node)
        candidates.remove(best_node)
        covered.update(best_new_covered)

    return selected, len(covered)


def greedy_max_cover_value(
    graph: nx.Graph,
    k: int,
    cover_radius: int = 1,
    normalize: bool = True,
) -> float:
    """Return greedy maximum coverage value for a graph.

    When ``normalize`` is true, the objective is divided by the number of nodes
    in the sampled graph, producing labels in ``[0, 1]``.
    """
    _, value = greedy_max_cover(graph, k=k, cover_radius=cover_radius)
    if normalize:
        n = graph.number_of_nodes()
        return 0.0 if n == 0 else value / n
    return float(value)

