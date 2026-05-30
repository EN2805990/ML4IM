from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .data import (
    StochasticMaxCoverDataset,
    build_synthetic_graph_dataset,
    collate_scenario_graphs,
)
from .model import ScenarioGNNRegressor


@dataclass
class TrainResult:
    best_val_mae: float
    final_test_mae: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def evaluate(
    model: ScenarioGNNRegressor,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_abs = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            pred = model(
                batch["x"],
                batch["edge_index"],
                batch["subgraph_batch"],
                batch["subgraph_group"],
            )
            target = batch["targets"]
            loss = F.mse_loss(pred, target, reduction="sum")
            total_loss += float(loss.item())
            total_abs += float(torch.abs(pred - target).sum().item())
            total_count += int(target.numel())

    return {
        "mse": total_loss / max(total_count, 1),
        "mae": total_abs / max(total_count, 1),
    }


def train_on_dataset(
    dataset: StochasticMaxCoverDataset,
    args: argparse.Namespace,
) -> TrainResult:
    seed_everything(args.seed)
    device = torch.device(args.device)

    n_total = len(dataset)
    if n_total < 3:
        raise ValueError("training requires at least 3 graphs for train/val/test splits")

    n_train = max(1, int(0.7 * n_total))
    n_val = max(1, int(0.15 * n_total))
    n_test = n_total - n_train - n_val
    while n_test <= 0 and n_train > 1:
        n_train -= 1
        n_test += 1

    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set, test_set = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=generator,
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "collate_fn": collate_scenario_graphs,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    model = ScenarioGNNRegressor(
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
    best_val_mae = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_count = 0

        progress = tqdm(
            train_loader,
            desc=f"epoch {epoch:03d}",
            disable=args.no_progress,
        )
        for batch in progress:
            batch = move_batch_to_device(batch, device)
            pred = model(
                batch["x"],
                batch["edge_index"],
                batch["subgraph_batch"],
                batch["subgraph_group"],
            )
            target = batch["targets"]
            loss = F.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            running_loss += float(loss.item()) * int(target.numel())
            running_count += int(target.numel())
            progress.set_postfix(loss=running_loss / max(running_count, 1))

        val_metrics = evaluate(model, val_loader, device)
        train_mse = running_loss / max(running_count, 1)

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        print(
            f"epoch={epoch:03d} "
            f"train_mse={train_mse:.5f} "
            f"val_mse={val_metrics['mse']:.5f} "
            f"val_mae={val_metrics['mae']:.5f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, device)
    print(
        f"best_val_mae={best_val_mae:.5f} "
        f"test_mse={test_metrics['mse']:.5f} "
        f"test_mae={test_metrics['mae']:.5f}"
    )

    if args.checkpoint:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "best_val_mae": best_val_mae,
                "test_metrics": test_metrics,
            },
            args.checkpoint,
        )
        print(f"saved checkpoint to {args.checkpoint}")

    return TrainResult(
        best_val_mae=best_val_mae,
        final_test_mae=test_metrics["mae"],
    )


def train_on_graph_dataset(
    graph_dataset,
    args: argparse.Namespace,
) -> TrainResult:
    dataset = StochasticMaxCoverDataset(
        graph_dataset=graph_dataset,
        m_samples=args.m_samples,
        k=args.k,
        cover_radius=args.cover_radius,
        node_keep_prob=args.node_keep_prob,
        edge_keep_prob=args.edge_keep_prob,
        seed=args.seed,
        scenario_dir=args.scenario_dir or None,
        load_saved=args.load_saved_scenarios,
        save_generated=args.save_scenarios_during_train,
        saved_sample_cache_size=args.saved_sample_cache_size,
    )
    return train_on_dataset(dataset, args)


def train(args: argparse.Namespace) -> TrainResult:
    graphs = []
    if not args.load_saved_scenarios:
        graphs = build_synthetic_graph_dataset(
            num_graphs=args.num_graphs,
            min_nodes=args.min_nodes,
            max_nodes=args.max_nodes,
            edge_prob=args.edge_prob,
            seed=args.seed,
        )
    return train_on_graph_dataset(graphs, args)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a GNN regressor for stochastic maximum coverage."
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
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--scenario-dir", type=str, default="")
    parser.add_argument("--load-saved-scenarios", action="store_true")
    parser.add_argument("--save-scenarios-during-train", action="store_true")
    parser.add_argument("--saved-sample-cache-size", type=int, default=128)
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
