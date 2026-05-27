import torch
from torch.utils.data import DataLoader

from smc_ml4co.data import (
    StochasticMaxCoverDataset,
    build_synthetic_graph_dataset,
    collate_flattened_policy_scenarios,
    collate_scenario_graphs,
    collate_policy_scenarios,
    load_scenario_samples,
)
from smc_ml4co.model import NodePolicyGNN, ScenarioGNNRegressor


def test_dataset_target_is_mean_of_subgraph_values():
    graphs = build_synthetic_graph_dataset(num_graphs=3, min_nodes=8, max_nodes=10, seed=1)
    dataset = StochasticMaxCoverDataset(graphs, m_samples=4, k=2, seed=1)

    sample = dataset[0]
    expected = sum(sample.subgraph_values) / len(sample.subgraph_values)
    assert abs(sample.target - expected) < 1e-8


def test_model_forward_shape():
    graphs = build_synthetic_graph_dataset(num_graphs=4, min_nodes=8, max_nodes=12, seed=2)
    dataset = StochasticMaxCoverDataset(graphs, m_samples=3, k=2, seed=2)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_scenario_graphs)
    batch = next(iter(loader))

    model = ScenarioGNNRegressor(hidden_dim=16, num_layers=2)
    pred = model(
        batch["x"],
        batch["edge_index"],
        batch["subgraph_batch"],
        batch["subgraph_group"],
    )

    assert pred.shape == torch.Size([2])


def test_scenarios_are_saved_and_loaded(tmp_path):
    graphs = build_synthetic_graph_dataset(num_graphs=2, min_nodes=8, max_nodes=10, seed=3)
    StochasticMaxCoverDataset(
        graphs,
        m_samples=5,
        k=2,
        seed=3,
        compute_labels=False,
        scenario_dir=tmp_path,
    )

    loaded = load_scenario_samples(tmp_path)
    assert len(loaded) == 2
    assert len(loaded[0].subgraphs) == 5


def test_unsupervised_policy_forward_shape():
    graphs = build_synthetic_graph_dataset(num_graphs=4, min_nodes=8, max_nodes=12, seed=4)
    dataset = StochasticMaxCoverDataset(
        graphs,
        m_samples=3,
        k=2,
        seed=4,
        compute_labels=False,
    )
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_policy_scenarios)
    batch = next(iter(loader))

    model = NodePolicyGNN(hidden_dim=16, num_layers=2)
    logits = model(batch["x"], batch["edge_index"])

    assert logits.shape == torch.Size([batch["x"].size(0)])


def test_flattened_policy_collate_expands_all_subgraphs():
    graphs = build_synthetic_graph_dataset(num_graphs=3, min_nodes=8, max_nodes=12, seed=5)
    dataset = StochasticMaxCoverDataset(
        graphs,
        m_samples=4,
        k=2,
        seed=5,
        compute_labels=False,
    )
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_flattened_policy_scenarios)
    batch = next(iter(loader))

    assert len(batch["subgraphs"]) == 2 * 4
    assert batch["subgraph_node_offsets"].shape == torch.Size([2 * 4])
    assert batch["subgraph_node_counts"].shape == torch.Size([2 * 4])
