from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .data import (
    StochasticMaxCoverDataset,
    build_synthetic_graph_dataset,
    collate_flattened_policy_scenarios,
)
from .model import NodePolicyGNN
from .objective import covered_nodes


@dataclass
class UnsupervisedTrainResult:
    best_val_reward: float
    final_test_reward: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_tensor_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def coverage_neighborhoods(subgraph: nx.Graph, cover_radius: int) -> list[list[int]]:
    neighborhoods: list[list[int]] = []
    for node in subgraph.nodes():
        lengths = nx.single_source_shortest_path_length(
            subgraph,
            node,
            cutoff=cover_radius,
        )
        neighborhoods.append(list(lengths.keys()))
    return neighborhoods


def soft_subgraph_reward(
    node_probs: torch.Tensor,
    subgraph: nx.Graph,
    cover_radius: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Differentiable expected coverage for one sampled subgraph."""
    if subgraph.number_of_nodes() == 0:
        return node_probs.new_tensor(0.0)

    subgraph_nodes = list(subgraph.nodes())
    node_to_local = {node: i for i, node in enumerate(subgraph_nodes)}
    covered_terms: list[torch.Tensor] = []

    for neighborhood in coverage_neighborhoods(subgraph, cover_radius):
        local_ids = [node_to_local[node] for node in neighborhood if node in node_to_local]
        if not local_ids:
            covered_terms.append(node_probs.new_tensor(0.0))
            continue

        idx = torch.tensor(local_ids, dtype=torch.long, device=node_probs.device)
        probs = node_probs.index_select(0, idx).clamp(eps, 1.0 - eps)
        not_covered = torch.prod(1.0 - probs)
        covered_terms.append(1.0 - not_covered)

    return torch.stack(covered_terms).mean()


def unsupervised_batch_loss(
    logits: torch.Tensor,
    batch: dict,
    k: int,
    cover_radius: int,
    temperature: float,
    budget_penalty_weight: float,
    entropy_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    rewards: list[torch.Tensor] = []
    budget_penalties: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []

    for subgraph_id, subgraph in enumerate(batch["subgraphs"]):
        offset = int(batch["subgraph_node_offsets"][subgraph_id].item())
        count = int(batch["subgraph_node_counts"][subgraph_id].item())
        subgraph_logits = logits[offset:offset + count]
        node_probs = torch.sigmoid(subgraph_logits / temperature)

        rewards.append(
            soft_subgraph_reward(
                node_probs=node_probs,
                subgraph=subgraph,
                cover_radius=cover_radius,
            )
        )

        budget_overflow = torch.relu(node_probs.sum() - float(k))
        budget_penalties.append(budget_overflow.pow(2) / max(float(k), 1.0))

        entropy = -(
            node_probs * torch.log(node_probs.clamp_min(1e-6))
            + (1.0 - node_probs) * torch.log((1.0 - node_probs).clamp_min(1e-6))
        ).mean()
        entropies.append(entropy)

    reward = torch.stack(rewards).mean()
    budget_penalty = torch.stack(budget_penalties).mean()
    entropy = torch.stack(entropies).mean()
    loss = -reward + budget_penalty_weight * budget_penalty - entropy_weight * entropy

    metrics = {
        "reward": float(reward.detach().cpu().item()),
        "budget_penalty": float(budget_penalty.detach().cpu().item()),
        "entropy": float(entropy.detach().cpu().item()),
    }
    return loss, metrics


def hard_topk_subgraph_reward(
    logits: torch.Tensor,
    subgraph: nx.Graph,
    k: int,
    cover_radius: int,
) -> float:
    subgraph_nodes = list(subgraph.nodes())
    if not subgraph_nodes or k <= 0:
        return 0.0

    selected_count = min(k, len(subgraph_nodes))
    topk = torch.topk(logits.detach().cpu(), k=selected_count).indices.tolist()
    selected_nodes = [subgraph_nodes[idx] for idx in topk]
    value = len(covered_nodes(subgraph, selected_nodes, cover_radius=cover_radius))
    return value / subgraph.number_of_nodes()


def evaluate_policy(
    model: NodePolicyGNN,
    loader: DataLoader,
    device: torch.device,
    k: int,
    cover_radius: int,
) -> dict[str, float]:
    model.eval()
    rewards: list[float] = []

    with torch.no_grad():
        for batch in loader:
            batch = move_tensor_batch_to_device(batch, device)
            logits = model(batch["x"], batch["edge_index"])

            for subgraph_id, subgraph in enumerate(batch["subgraphs"]):
                offset = int(batch["subgraph_node_offsets"][subgraph_id].item())
                count = int(batch["subgraph_node_counts"][subgraph_id].item())
                subgraph_logits = logits[offset:offset + count]
                rewards.append(
                    hard_topk_subgraph_reward(
                        subgraph_logits,
                        subgraph=subgraph,
                        k=k,
                        cover_radius=cover_radius,
                    )
                )

    return {"reward": float(np.mean(rewards)) if rewards else 0.0}


def split_dataset(dataset: StochasticMaxCoverDataset, seed: int):
    n_total = len(dataset)
    if n_total < 3:
        raise ValueError("training requires at least 3 graphs for train/val/test splits")

    n_train = max(1, int(0.7 * n_total))
    n_val = max(1, int(0.15 * n_total))
    n_test = n_total - n_train - n_val
    while n_test <= 0 and n_train > 1:
        n_train -= 1
        n_test += 1

    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val, n_test], generator=generator)


def train_unsupervised_on_dataset(
    dataset: StochasticMaxCoverDataset,
    args: argparse.Namespace,
) -> UnsupervisedTrainResult:
    seed_everything(args.seed)
    device = torch.device(args.device)

    train_set, val_set, test_set = split_dataset(dataset, args.seed)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "collate_fn": collate_flattened_policy_scenarios,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    model = NodePolicyGNN(
        in_dim=4,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_state = None
    best_val_reward = -float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_reward = 0.0
        running_count = 0

        progress = tqdm(
            train_loader,
            desc=f"unsup epoch {epoch:03d}",
            disable=args.no_progress,
        )
        for batch in progress:
            batch = move_tensor_batch_to_device(batch, device)
            logits = model(batch["x"], batch["edge_index"])
            loss, metrics = unsupervised_batch_loss(
                logits=logits,
                batch=batch,
                k=args.k,
                cover_radius=args.cover_radius,
                temperature=args.temperature,
                budget_penalty_weight=args.budget_penalty_weight,
                entropy_weight=args.entropy_weight,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            batch_count = len(batch["subgraphs"])
            running_loss += float(loss.item()) * batch_count
            running_reward += metrics["reward"] * batch_count
            running_count += batch_count
            progress.set_postfix(
                loss=running_loss / max(running_count, 1),
                reward=running_reward / max(running_count, 1),
            )

        val_metrics = evaluate_policy(
            model,
            val_loader,
            device=device,
            k=args.k,
            cover_radius=args.cover_radius,
        )
        train_loss = running_loss / max(running_count, 1)
        train_reward = running_reward / max(running_count, 1)

        if val_metrics["reward"] > best_val_reward:
            best_val_reward = val_metrics["reward"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.5f} "
            f"train_soft_reward={train_reward:.5f} "
            f"val_hard_reward={val_metrics['reward']:.5f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_policy(
        model,
        test_loader,
        device=device,
        k=args.k,
        cover_radius=args.cover_radius,
    )
    print(
        f"best_val_hard_reward={best_val_reward:.5f} "
        f"test_hard_reward={test_metrics['reward']:.5f}"
    )

    if args.checkpoint:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "best_val_reward": best_val_reward,
                "test_metrics": test_metrics,
            },
            args.checkpoint,
        )
        print(f"saved checkpoint to {args.checkpoint}")

    return UnsupervisedTrainResult(
        best_val_reward=best_val_reward,
        final_test_reward=test_metrics["reward"],
    )


def train_unsupervised_on_graph_dataset(
    graph_dataset,
    args: argparse.Namespace,
) -> UnsupervisedTrainResult:
    dataset = StochasticMaxCoverDataset(
        graph_dataset=graph_dataset,
        m_samples=args.m_samples,
        k=args.k,
        cover_radius=args.cover_radius,
        node_keep_prob=args.node_keep_prob,
        edge_keep_prob=args.edge_keep_prob,
        seed=args.seed,
        compute_labels=False,
        scenario_dir=args.scenario_dir or None,
        load_saved=args.load_saved_scenarios,
    )
    return train_unsupervised_on_dataset(dataset, args)


def train(args: argparse.Namespace) -> UnsupervisedTrainResult:
    graphs = build_synthetic_graph_dataset(
        num_graphs=args.num_graphs,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        edge_prob=args.edge_prob,
        seed=args.seed,
    )
    return train_unsupervised_on_graph_dataset(graphs, args)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unsupervised ML4CO training for stochastic maximum coverage."
    )
    parser.add_argument("--num-graphs", type=int, default=100)
    parser.add_argument("--min-nodes", type=int, default=20)
    parser.add_argument("--max-nodes", type=int, default=80)
    parser.add_argument("--edge-prob", type=float, default=0.08)
    parser.add_argument("--m-samples", type=int, default=8)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--cover-radius", type=int, default=1)
    parser.add_argument("--node-keep-prob", type=float, default=0.8)
    parser.add_argument("--edge-keep-prob", type=float, default=0.9)
    parser.add_argument("--scenario-dir", type=str, default="scenario_cache")
    parser.add_argument("--load-saved-scenarios", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--budget-penalty-weight", type=float, default=2.0)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
