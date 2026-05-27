from __future__ import annotations

import torch
from torch import nn


def global_mean_pool(x: torch.Tensor, batch: torch.Tensor, num_groups: int) -> torch.Tensor:
    if x.numel() == 0:
        return x.new_zeros((num_groups, x.size(-1)))

    out = x.new_zeros((num_groups, x.size(-1)))
    out.index_add_(0, batch, x)

    counts = x.new_zeros((num_groups, 1))
    ones = x.new_ones((x.size(0), 1))
    counts.index_add_(0, batch, ones)
    return out / counts.clamp_min(1.0)


class GCNLayer(nn.Module):
    """Small dependency-free GCN layer using edge_index aggregation."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        if edge_index.numel() == 0:
            return h

        src, dst = edge_index
        deg = torch.bincount(dst, minlength=x.size(0)).to(h.dtype).clamp_min(1.0)
        agg = h.new_zeros(h.shape)
        agg.index_add_(0, dst, h[src])
        return agg / deg.unsqueeze(-1)


class ScenarioGNNRegressor(nn.Module):
    """Encode sampled subgraphs and average them into original-graph outputs."""

    def __init__(
        self,
        in_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        layers = []
        last_dim = in_dim
        for _ in range(num_layers):
            layers.append(GCNLayer(last_dim, hidden_dim))
            last_dim = hidden_dim
        self.layers = nn.ModuleList(layers)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        subgraph_batch: torch.Tensor,
        subgraph_group: torch.Tensor,
    ) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, edge_index)
            h = self.activation(h)
            h = self.dropout(h)

        num_subgraphs = int(subgraph_group.numel())
        subgraph_emb = global_mean_pool(h, subgraph_batch, num_subgraphs)

        num_graphs = int(subgraph_group.max().item()) + 1 if num_subgraphs else 0
        graph_emb = global_mean_pool(subgraph_emb, subgraph_group, num_graphs)
        pred = self.head(graph_emb).squeeze(-1)
        return pred


class NodePolicyGNN(nn.Module):
    """Node-level policy network for unsupervised maximum coverage training."""

    def __init__(
        self,
        in_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        layers = []
        last_dim = in_dim
        for _ in range(num_layers):
            layers.append(GCNLayer(last_dim, hidden_dim))
            last_dim = hidden_dim

        self.layers = nn.ModuleList(layers)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, edge_index)
            h = self.activation(h)
            h = self.dropout(h)
        return self.head(h).squeeze(-1)
