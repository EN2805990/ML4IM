from smc_ml4co.unsupervised import build_arg_parser, train


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--num-graphs", "60",
            "--min-nodes", "20",
            "--max-nodes", "50",
            "--m-samples", "6",
            "--epochs", "5",
            "--batch-size", "6",
            "--scenario-dir", "scenario_cache",
            "--no-progress",
        ]
    )
    train(args)


if __name__ == "__main__":
    main()
