"""Stream Wikipedia articles into a text file of a requested minimum size."""

from __future__ import annotations

import argparse
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Mapping


DEFAULT_TARGET_CHARACTERS = 1_000_000
DATASET_NAME = "wikimedia/wikipedia"
DATASET_CONFIG = "20231101.en"
COUNT_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)\s*([kmb]?)$", re.IGNORECASE)
MULTIPLIERS = {
    "": 1,
    "k": 1_000,
    "m": 1_000_000,
    "b": 1_000_000_000,
}


def parse_character_count(value: str) -> int:
    """Parse counts such as ``1500000``, ``1.5m``, or ``10m``."""
    normalized = value.replace(",", "").replace("_", "").strip()
    match = COUNT_PATTERN.fullmatch(normalized)
    if match is None:
        raise argparse.ArgumentTypeError(
            "character count must be a positive number, optionally ending in k, m, or b"
        )

    number_text, suffix = match.groups()
    try:
        characters = Decimal(number_text) * MULTIPLIERS[suffix.lower()]
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError("invalid character count") from error

    if characters <= 0 or characters != characters.to_integral_value():
        raise argparse.ArgumentTypeError("character count must resolve to a positive integer")

    return int(characters)


def compact_character_count(characters: int) -> str:
    """Format a count for filenames, for example 1_500_000 as ``1.5m``."""
    for suffix, multiplier in (("b", 1_000_000_000), ("m", 1_000_000), ("k", 1_000)):
        if characters >= multiplier:
            value = Decimal(characters) / multiplier
            label = format(value.normalize(), "f")
            return f"{label}{suffix}"
    return str(characters)


def default_output_path(target_characters: int) -> Path:
    """Build an output filename that reflects the requested character count."""
    label = compact_character_count(target_characters)
    return Path("raw_data") / f"wikipedia_{label}_chars.txt"


def write_articles(
    articles: Iterable[Mapping[str, str]],
    output_path: Path,
    target_characters: int,
) -> tuple[int, int]:
    """Write complete articles until the file reaches the requested size."""
    notice = (
        "Content sourced from English Wikipedia.\n"
        "Licenses: CC BY-SA 3.0 and GFDL.\n"
        "Article titles and source URLs are included for attribution.\n"
    )
    characters_written = len(notice)
    article_count = 0

    with output_path.open("w", encoding="utf-8") as output:
        output.write(notice)

        for article in articles:
            block = (
                f"\n\n{'=' * 80}\n"
                f"TITLE: {article['title']}\n"
                f"SOURCE: {article['url']}\n"
                f"{'=' * 80}\n\n"
                f"{article['text']}\n"
            )
            output.write(block)
            characters_written += len(block)
            article_count += 1

            if characters_written >= target_characters:
                break

    if characters_written < target_characters:
        raise RuntimeError(
            f"dataset ended after {characters_written:,} characters, before the "
            f"{target_characters:,}-character target"
        )

    return article_count, characters_written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a text file from streamed English Wikipedia articles."
    )
    parser.add_argument(
        "--target-characters",
        type=parse_character_count,
        default=DEFAULT_TARGET_CHARACTERS,
        metavar="COUNT",
        help="minimum output size; accepts values such as 1500000, 1.5m, or 10m",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output path (default: raw_data/wikipedia_<target>_chars.txt)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_path = args.output or default_output_path(args.target_characters)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit(
            "Missing dependency: install it with `python -m pip install datasets`."
        ) from error

    articles = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split="train",
        streaming=True,
    )
    article_count, characters_written = write_articles(
        articles,
        output_path,
        args.target_characters,
    )

    print(f"Created: {output_path}")
    print(f"Articles: {article_count:,}")
    print(f"Characters: {characters_written:,}")


if __name__ == "__main__":
    main()
