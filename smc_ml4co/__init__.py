"""Stochastic maximum coverage ML4CO utilities."""

__all__ = [
    "StochasticMaxCoverDataset",
    "build_scenario_samples",
    "build_synthetic_graph_dataset",
    "greedy_max_cover_value",
    "load_scenario_samples",
    "preprocess_graph_dataset",
    "save_scenario_samples",
]


def __getattr__(name: str):
    if name in {
        "StochasticMaxCoverDataset",
        "build_scenario_samples",
        "build_synthetic_graph_dataset",
        "load_scenario_samples",
        "save_scenario_samples",
    }:
        from .data import (
            StochasticMaxCoverDataset,
            build_scenario_samples,
            build_synthetic_graph_dataset,
            load_scenario_samples,
            save_scenario_samples,
        )

        return {
            "StochasticMaxCoverDataset": StochasticMaxCoverDataset,
            "build_scenario_samples": build_scenario_samples,
            "build_synthetic_graph_dataset": build_synthetic_graph_dataset,
            "load_scenario_samples": load_scenario_samples,
            "save_scenario_samples": save_scenario_samples,
        }[name]

    if name == "preprocess_graph_dataset":
        from .preprocess import preprocess_graph_dataset

        return preprocess_graph_dataset

    if name == "greedy_max_cover_value":
        from .objective import greedy_max_cover_value

        return greedy_max_cover_value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
