"""Command-line entry point for training and dataset preparation."""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.toml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train BDH or download a Wikipedia text dataset."
    )
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument(
        "--train",
        action="store_true",
        help="train BDH using the configuration in config.toml",
    )
    actions.add_argument(
        "--download-wikipedia",
        action="store_true",
        help="download and assemble an English Wikipedia text dataset",
    )
    actions.add_argument(
        "--train-tokenizer",
        action="store_true",
        help="train a tokenizer using the [tokenizer] configuration",
    )
    actions.add_argument(
        "--tokenize-dataset",
        action="store_true",
        help="tokenize the configured dataset using the configured tokenizer",
    )
    actions.add_argument(
        "--full",
        action="store_true",
        help="train the tokenizer, tokenize the dataset, and train BDH",
    )
    parser.add_argument(
        "--target-characters",
        help="Wikipedia output size, such as 1.5m or 10m",
    )
    parser.add_argument(
        "--output",
        help="Wikipedia output path",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="configuration TOML file (default: config.toml)",
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="check BDH setup and report model size without training (requires --train or --full)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.train or args.train_tokenizer or args.tokenize_dataset or args.full:
        if args.target_characters or args.output:
            raise SystemExit(
                "--target-characters and --output require --download-wikipedia"
            )

        if args.dry and not (args.train or args.full):
            raise SystemExit("--dry requires --train or --full")

        if args.full:
            from src.train_tokenizer import main as train_tokenizer_main

            print("Stage 1/3: Training tokenizer")
            train_tokenizer_main(args.config)

            from src.tokenize_dataset import main as tokenize_dataset_main

            print("Stage 2/3: Tokenizing dataset")
            tokenize_dataset_main(["--config", str(args.config)])

            from src.train import main as train_main

            print("Stage 3/3: Training model")
            train_main(dry=args.dry, config_path=args.config)
        elif args.train:
            from src.train import main as train_main

            train_main(dry=args.dry, config_path=args.config)
        elif args.train_tokenizer:
            from src.train_tokenizer import main as train_tokenizer_main

            train_tokenizer_main(args.config)
        else:
            from src.tokenize_dataset import main as tokenize_dataset_main

            # The wrapper flag has already been consumed by this parser.
            tokenize_dataset_main(["--config", str(args.config)])
        return

    from src.data import wikipedia_dataset

    wikipedia_args = []
    if args.target_characters:
        wikipedia_args.extend(["--target-characters", args.target_characters])
    if args.output:
        wikipedia_args.extend(["--output", args.output])
    wikipedia_dataset.main(wikipedia_args)


if __name__ == "__main__":
    main()
