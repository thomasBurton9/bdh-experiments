"""Command-line entry point for training and dataset preparation."""

from __future__ import annotations

import argparse


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
    parser.add_argument(
        "--target-characters",
        help="Wikipedia output size, such as 1.5m or 10m",
    )
    parser.add_argument(
        "--output",
        help="Wikipedia output path",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.train:
        if args.target_characters or args.output:
            raise SystemExit("--target-characters and --output require --download-wikipedia")
        from src.train import main as train_main

        train_main()
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
