from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Sequence

import networkx as nx

from .data import (
    GraphScenarioSample,
    build_scenario_samples,
    build_synthetic_graph_dataset,
    save_scenario_samples,
)


def load_graph_dataset_from_pickle(path: str | Path) -> list[nx.Graph]:
    """Load a list[networkx.Graph] from a pickle file."""
    input_path = Path(path)
    with input_path.open("rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, nx.Graph):
        return [payload]

    if not isinstance(payload, Sequence):
        raise TypeError(f"expected a Sequence[nx.Graph], got {type(payload)!r}")

    graphs: list[nx.Graph] = []
    for i, graph in enumerate(payload):
        if not isinstance(graph, nx.Graph):
            raise TypeError(f"payload[{i}] is not a networkx.Graph: {type(graph)!r}")
        graphs.append(graph)

    return graphs


def preprocess_graph_dataset(
    graph_dataset: Sequence[nx.Graph],
    scenario_dir: str | Path,
    m_samples: int,
    k: int,
    cover_radius: int = 1,
    node_keep_prob: float = 0.8,
    edge_keep_prob: float = 1.0,
    seed: int = 0,
    compute_labels: bool = True,
) -> list[GraphScenarioSample]:
    """Run standalone dataset processing: sample subgraphs and save offline files."""
    samples = build_scenario_samples(
        graph_dataset=graph_dataset,
        m_samples=m_samples,
        k=k,
        cover_radius=cover_radius,
        node_keep_prob=node_keep_prob,
        edge_keep_prob=edge_keep_prob,
        seed=seed,
        compute_labels=compute_labels,
    )
    save_scenario_samples(samples, scenario_dir=scenario_dir)
    return samples


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone dataset preprocessing for stochastic max cover."
    )
    parser.add_argument(
        "--graphs-pkl",
        type=str,
        default="",
        help="Path to pickle containing Sequence[networkx.Graph]. If empty, synthetic graphs are generated.",
    )
    parser.add_argument("--scenario-dir", type=str, required=True)
    parser.add_argument("--m-samples", type=int, default=8)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--cover-radius", type=int, default=1)
    parser.add_argument("--node-keep-prob", type=float, default=0.8)
    parser.add_argument("--edge-keep-prob", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--without-labels", action="store_true")

    # Synthetic-only options (used when --graphs-pkl is empty).
    parser.add_argument("--num-graphs", type=int, default=100)
    parser.add_argument("--min-nodes", type=int, default=20)
    parser.add_argument("--max-nodes", type=int, default=80)
    parser.add_argument("--edge-prob", type=float, default=0.08)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.graphs_pkl:
        graphs = load_graph_dataset_from_pickle(args.graphs_pkl)
    else:
        graphs = build_synthetic_graph_dataset(
            num_graphs=args.num_graphs,
            min_nodes=args.min_nodes,
            max_nodes=args.max_nodes,
            edge_prob=args.edge_prob,
            seed=args.seed,
        )

    samples = preprocess_graph_dataset(
        graph_dataset=graphs,
        scenario_dir=args.scenario_dir,
        m_samples=args.m_samples,
        k=args.k,
        cover_radius=args.cover_radius,
        node_keep_prob=args.node_keep_prob,
        edge_keep_prob=args.edge_keep_prob,
        seed=args.seed,
        compute_labels=not args.without_labels,
    )

    print(
        f"processed_graphs={len(samples)} "
        f"m_samples={args.m_samples} "
        f"scenario_dir={args.scenario_dir}"
    )


if __name__ == "__main__":
    main()
