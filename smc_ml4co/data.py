from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset

from .objective import greedy_max_cover_value

ObjectiveFn = Callable[[nx.Graph], float]


@dataclass(frozen=True)
class GraphScenarioSample:
    graph_id: int
    original_graph: nx.Graph
    subgraphs: list[nx.Graph]
    subgraph_values: list[float]
    target: float


def relabel_graph_to_int(graph: nx.Graph) -> nx.Graph:
    """Return a copy whose node ids are contiguous ints starting from zero."""
    mapping = {node: i for i, node in enumerate(graph.nodes())}
    return nx.relabel_nodes(graph, mapping, copy=True)


def sample_subgraph(
    graph: nx.Graph,
    rng: np.random.Generator,
    node_keep_prob: float = 0.8,
    edge_keep_prob: float = 1.0,
    keep_at_least_one_node: bool = True,
) -> nx.Graph:
    """Sample a stochastic subgraph using node dropout and edge dropout."""
    if not 0.0 <= node_keep_prob <= 1.0:
        raise ValueError("node_keep_prob must be in [0, 1]")
    if not 0.0 <= edge_keep_prob <= 1.0:
        raise ValueError("edge_keep_prob must be in [0, 1]")

    nodes = list(graph.nodes())
    kept_nodes = [node for node in nodes if rng.random() <= node_keep_prob]

    if keep_at_least_one_node and nodes and not kept_nodes:
        kept_nodes = [nodes[int(rng.integers(0, len(nodes)))]]

    subgraph = graph.subgraph(kept_nodes).copy()

    if edge_keep_prob < 1.0:
        remove_edges = [
            edge for edge in subgraph.edges()
            if rng.random() > edge_keep_prob
        ]
        subgraph.remove_edges_from(remove_edges)

    return subgraph


def graph_node_features(graph: nx.Graph) -> torch.Tensor:
    """Create simple structural node features for a NetworkX graph."""
    n = graph.number_of_nodes()
    if n == 0:
        return torch.empty((0, 4), dtype=torch.float32)

    degrees = np.array([graph.degree(node) for node in graph.nodes()], dtype=np.float32)
    max_degree = max(float(degrees.max()), 1.0)

    clustering_by_node = nx.clustering(graph)
    clustering = np.array(
        [clustering_by_node[node] for node in graph.nodes()],
        dtype=np.float32,
    )

    features = np.stack(
        [
            np.ones(n, dtype=np.float32),
            degrees / max_degree,
            degrees / max(float(n - 1), 1.0),
            clustering,
        ],
        axis=1,
    )
    return torch.from_numpy(features)


def graph_edge_index(graph: nx.Graph) -> torch.Tensor:
    """Return a directed edge_index tensor with self-loops."""
    nodes = list(graph.nodes())
    node_to_local = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    edges: list[tuple[int, int]] = []

    for u, v in graph.edges():
        u_local = node_to_local[u]
        v_local = node_to_local[v]
        edges.append((u_local, v_local))
        edges.append((v_local, u_local))

    for node in range(n):
        edges.append((node, node))

    if not edges:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


class StochasticMaxCoverDataset(Dataset):
    """Dataset that groups m sampled subgraphs under each original graph."""

    def __init__(
        self,
        graph_dataset: Sequence[nx.Graph],
        m_samples: int,
        k: int,
        cover_radius: int = 1,
        node_keep_prob: float = 0.8,
        edge_keep_prob: float = 1.0,
        seed: int = 0,
        objective_fn: ObjectiveFn | None = None,
        compute_labels: bool = True,
        scenario_dir: str | Path | None = None,
        load_saved: bool = False,
        save_generated: bool = False,
    ) -> None:
        if m_samples <= 0:
            raise ValueError("m_samples must be positive")

        self.graph_dataset = [
            relabel_graph_to_int(graph) for graph in graph_dataset
        ]
        self.m_samples = m_samples
        self.k = k
        self.cover_radius = cover_radius
        self.node_keep_prob = node_keep_prob
        self.edge_keep_prob = edge_keep_prob
        self.seed = seed
        self.compute_labels = compute_labels
        self.scenario_dir = Path(scenario_dir) if scenario_dir else None
        self.save_generated = save_generated
        self.objective_fn = objective_fn or (
            lambda graph: greedy_max_cover_value(
                graph,
                k=k,
                cover_radius=cover_radius,
                normalize=True,
            )
        )

        if load_saved:
            if self.scenario_dir is None:
                raise ValueError("scenario_dir is required when load_saved=True")
            self.samples = load_scenario_samples(self.scenario_dir)
        else:
            self.samples = build_scenario_samples(
                graph_dataset=self.graph_dataset,
                m_samples=self.m_samples,
                k=self.k,
                cover_radius=self.cover_radius,
                node_keep_prob=self.node_keep_prob,
                edge_keep_prob=self.edge_keep_prob,
                seed=self.seed,
                objective_fn=self.objective_fn,
                compute_labels=self.compute_labels,
            )
            if self.save_generated and self.scenario_dir is not None:
                save_scenario_samples(self.samples, self.scenario_dir)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> GraphScenarioSample:
        return self.samples[index]


def save_scenario_samples(
    samples: Sequence[GraphScenarioSample],
    scenario_dir: str | Path,
) -> None:
    """Persist every sampled subgraph for later supervised or unsupervised use."""
    output_dir = Path(scenario_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "num_graphs": len(samples),
        "files": [],
    }

    for sample in samples:
        filename = f"graph_{sample.graph_id:06d}.pkl"
        payload = {
            "graph_id": sample.graph_id,
            "target": sample.target,
            "subgraph_values": sample.subgraph_values,
            "original_graph": sample.original_graph,
            "subgraphs": sample.subgraphs,
        }
        with (output_dir / filename).open("wb") as f:
            pickle.dump(payload, f)
        manifest["files"].append(filename)

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def build_scenario_samples(
    graph_dataset: Sequence[nx.Graph],
    m_samples: int,
    k: int,
    cover_radius: int = 1,
    node_keep_prob: float = 0.8,
    edge_keep_prob: float = 1.0,
    seed: int = 0,
    objective_fn: ObjectiveFn | None = None,
    compute_labels: bool = True,
) -> list[GraphScenarioSample]:
    """Sample stochastic subgraphs and build scenario samples in memory."""
    if m_samples <= 0:
        raise ValueError("m_samples must be positive")

    graphs = [relabel_graph_to_int(graph) for graph in graph_dataset]
    scenario_objective = objective_fn or (
        lambda graph: greedy_max_cover_value(
            graph,
            k=k,
            cover_radius=cover_radius,
            normalize=True,
        )
    )

    samples: list[GraphScenarioSample] = []
    for graph_id, graph in enumerate(graphs):
        rng = np.random.default_rng(seed + graph_id)
        subgraphs: list[nx.Graph] = []
        values: list[float] = []

        for _ in range(m_samples):
            subgraph = sample_subgraph(
                graph,
                rng=rng,
                node_keep_prob=node_keep_prob,
                edge_keep_prob=edge_keep_prob,
            )
            value = float(scenario_objective(subgraph)) if compute_labels else 0.0
            subgraphs.append(subgraph)
            values.append(value)

        target = float(np.mean(values)) if compute_labels else 0.0
        samples.append(
            GraphScenarioSample(
                graph_id=graph_id,
                original_graph=graph,
                subgraphs=subgraphs,
                subgraph_values=values,
                target=target,
            )
        )

    return samples


def load_scenario_samples(scenario_dir: str | Path) -> list[GraphScenarioSample]:
    """Load saved scenario samples produced by ``save_scenario_samples``."""
    input_dir = Path(scenario_dir)
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing scenario manifest: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    samples: list[GraphScenarioSample] = []
    for filename in manifest["files"]:
        with (input_dir / filename).open("rb") as f:
            payload = pickle.load(f)

        samples.append(
            GraphScenarioSample(
                graph_id=int(payload["graph_id"]),
                original_graph=payload["original_graph"],
                subgraphs=payload["subgraphs"],
                subgraph_values=[float(v) for v in payload.get("subgraph_values", [])],
                target=float(payload.get("target", 0.0)),
            )
        )

    return samples


def collate_scenario_graphs(samples: Sequence[GraphScenarioSample]) -> dict[str, torch.Tensor]:
    """Collate original-graph groups by flattening their sampled subgraphs."""
    x_parts: list[torch.Tensor] = []
    edge_parts: list[torch.Tensor] = []
    subgraph_batch_parts: list[torch.Tensor] = []
    graph_group_parts: list[torch.Tensor] = []
    subgraph_values: list[float] = []
    node_offset = 0
    subgraph_id = 0

    targets = torch.tensor([sample.target for sample in samples], dtype=torch.float32)
    graph_ids = torch.tensor([sample.graph_id for sample in samples], dtype=torch.long)

    for group_id, sample in enumerate(samples):
        for subgraph, value in zip(sample.subgraphs, sample.subgraph_values):
            x = graph_node_features(subgraph)
            edge_index = graph_edge_index(subgraph)

            if edge_index.numel() > 0:
                edge_parts.append(edge_index + node_offset)

            n = x.size(0)
            x_parts.append(x)
            subgraph_batch_parts.append(
                torch.full((n,), subgraph_id, dtype=torch.long)
            )
            graph_group_parts.append(
                torch.tensor([group_id], dtype=torch.long)
            )
            subgraph_values.append(float(value))

            node_offset += n
            subgraph_id += 1

    if x_parts:
        x_all = torch.cat(x_parts, dim=0)
        subgraph_batch = torch.cat(subgraph_batch_parts, dim=0)
    else:
        x_all = torch.empty((0, 4), dtype=torch.float32)
        subgraph_batch = torch.empty((0,), dtype=torch.long)

    if edge_parts:
        edge_index_all = torch.cat(edge_parts, dim=1)
    else:
        edge_index_all = torch.empty((2, 0), dtype=torch.long)

    return {
        "x": x_all,
        "edge_index": edge_index_all,
        "subgraph_batch": subgraph_batch,
        "subgraph_group": torch.cat(graph_group_parts, dim=0),
        "subgraph_values": torch.tensor(subgraph_values, dtype=torch.float32),
        "targets": targets,
        "graph_ids": graph_ids,
    }


def collate_policy_scenarios(samples: Sequence[GraphScenarioSample]) -> dict:
    """Collate original graphs while keeping saved subgraphs for objective loss."""
    x_parts: list[torch.Tensor] = []
    edge_parts: list[torch.Tensor] = []
    node_graph_parts: list[torch.Tensor] = []
    node_offsets: list[int] = []
    node_counts: list[int] = []
    node_offset = 0

    for group_id, sample in enumerate(samples):
        graph = sample.original_graph
        x = graph_node_features(graph)
        edge_index = graph_edge_index(graph)

        if edge_index.numel() > 0:
            edge_parts.append(edge_index + node_offset)

        n = x.size(0)
        x_parts.append(x)
        node_graph_parts.append(torch.full((n,), group_id, dtype=torch.long))
        node_offsets.append(node_offset)
        node_counts.append(n)
        node_offset += n

    x_all = torch.cat(x_parts, dim=0) if x_parts else torch.empty((0, 4), dtype=torch.float32)
    edge_index_all = (
        torch.cat(edge_parts, dim=1)
        if edge_parts
        else torch.empty((2, 0), dtype=torch.long)
    )
    node_graph_batch = (
        torch.cat(node_graph_parts, dim=0)
        if node_graph_parts
        else torch.empty((0,), dtype=torch.long)
    )

    return {
        "x": x_all,
        "edge_index": edge_index_all,
        "node_graph_batch": node_graph_batch,
        "node_offsets": torch.tensor(node_offsets, dtype=torch.long),
        "node_counts": torch.tensor(node_counts, dtype=torch.long),
        "samples": list(samples),
    }


def collate_flattened_policy_scenarios(samples: Sequence[GraphScenarioSample]) -> dict:
    """Collate sampled subgraphs directly for subgraph-level policy learning.

    This is used by the unsupervised pipeline when we want to run the policy
    network on every sampled subgraph (instead of only the original graph).
    All subgraphs in the mini-batch are flattened into one disconnected graph
    tensor, with offset metadata to recover each subgraph slice later.
    """
    x_parts: list[torch.Tensor] = []
    edge_parts: list[torch.Tensor] = []
    subgraph_batch_parts: list[torch.Tensor] = []
    graph_group_parts: list[torch.Tensor] = []
    subgraphs: list[nx.Graph] = []
    node_offsets: list[int] = []
    node_counts: list[int] = []
    node_offset = 0
    subgraph_id = 0

    for group_id, sample in enumerate(samples):
        for subgraph in sample.subgraphs:
            x = graph_node_features(subgraph)
            edge_index = graph_edge_index(subgraph)

            if edge_index.numel() > 0:
                # Shift local node ids so each subgraph occupies its own node range
                # inside the flattened batch tensor.
                edge_parts.append(edge_index + node_offset)

            n = x.size(0)
            x_parts.append(x)
            # Map every node to its flattened subgraph id (0..num_subgraphs-1).
            subgraph_batch_parts.append(torch.full((n,), subgraph_id, dtype=torch.long))
            # Record which original graph this subgraph belongs to.
            graph_group_parts.append(torch.tensor([group_id], dtype=torch.long))
            subgraphs.append(subgraph)
            # Save slice metadata so logits can be cut back per subgraph later:
            # logits[offset : offset + count].
            node_offsets.append(node_offset)
            node_counts.append(n)

            node_offset += n
            subgraph_id += 1

    x_all = torch.cat(x_parts, dim=0) if x_parts else torch.empty((0, 4), dtype=torch.float32)
    edge_index_all = (
        torch.cat(edge_parts, dim=1)
        if edge_parts
        else torch.empty((2, 0), dtype=torch.long)
    )
    subgraph_batch = (
        torch.cat(subgraph_batch_parts, dim=0)
        if subgraph_batch_parts
        else torch.empty((0,), dtype=torch.long)
    )
    subgraph_group = (
        torch.cat(graph_group_parts, dim=0)
        if graph_group_parts
        else torch.empty((0,), dtype=torch.long)
    )

    return {
        "x": x_all,
        "edge_index": edge_index_all,
        "subgraph_batch": subgraph_batch,
        "subgraph_group": subgraph_group,
        "subgraph_node_offsets": torch.tensor(node_offsets, dtype=torch.long),
        "subgraph_node_counts": torch.tensor(node_counts, dtype=torch.long),
        "subgraphs": subgraphs,
        "graph_ids": torch.tensor([sample.graph_id for sample in samples], dtype=torch.long),
    }


def build_synthetic_graph_dataset(
    num_graphs: int = 100,
    min_nodes: int = 20,
    max_nodes: int = 80,
    edge_prob: float = 0.08,
    seed: int = 0,
) -> list[nx.Graph]:
    """Create a mixed synthetic graph dataset for smoke tests and demos."""
    rng = np.random.default_rng(seed)
    graphs: list[nx.Graph] = []

    for i in range(num_graphs):
        n = int(rng.integers(min_nodes, max_nodes + 1))
        graph_seed = int(rng.integers(0, 2**31 - 1))

        if i % 3 == 0:
            graph = nx.erdos_renyi_graph(n=n, p=edge_prob, seed=graph_seed)
        elif i % 3 == 1:
            m = max(1, min(4, n - 1))
            graph = nx.barabasi_albert_graph(n=n, m=m, seed=graph_seed)
        else:
            k = max(2, min(n - 1, int(round(edge_prob * n))))
            if k % 2 == 1:
                k += 1
            if k >= n:
                k = n - 1 if (n - 1) % 2 == 0 else max(0, n - 2)
            if k <= 0:
                graph = nx.empty_graph(n)
                graphs.append(relabel_graph_to_int(graph))
                continue
            graph = nx.watts_strogatz_graph(n=n, k=k, p=0.25, seed=graph_seed)

        graphs.append(relabel_graph_to_int(graph))

    return graphs
