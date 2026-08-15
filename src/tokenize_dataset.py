"""Tokenize a text dataset and save its token IDs as a binary array."""

from __future__ import annotations

import argparse
import tomllib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.toml"


def configured_path(path: str) -> Path:
    """Resolve a configured path relative to the project root."""
    configured = Path(path)
    return configured if configured.is_absolute() else PROJECT_ROOT / configured


def default_paths() -> tuple[Path, Path]:
    """Read the tokenizer and dataset paths from the project configuration."""
    with CONFIG_PATH.open("rb") as config_file:
        config = tomllib.load(config_file)

    train_config = config.get("train", {})
    data_config = config.get("data", {})
    tokenizer_config = config.get("tokenizer", {})

    tokenizer_path = train_config.get(
        "tokenizer_path", tokenizer_config.get("OUTPUT_PATH", "tokenizers/tokenizer.json")
    )
    input_file_path = data_config.get("INPUT_FILE_PATH", "input.txt")
    return configured_path(tokenizer_path), configured_path(input_file_path)


def output_path_for(input_file_path: Path) -> Path:
    """Return the default tokenized-data path for an input file."""
    input_stem = input_file_path.stem
    return input_file_path.parent / input_stem / f"{input_stem}_tokenized"


def tokenize_dataset(
    tokenizer_path: Path, input_file_path: Path, output_path: Path | None = None
) -> Path:
    """Tokenize ``input_file_path`` and save raw uint32 token IDs."""
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    if not input_file_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file_path}")

    output_path = output_path or output_path_for(input_file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    text = input_file_path.read_text(encoding="utf-8")
    token_ids = np.asarray(tokenizer.encode(text).ids, dtype=np.uint32)
    token_ids.tofile(output_path)

    print(f"Tokenized {input_file_path} into {len(token_ids):,} tokens")
    print(f"Tokenized data saved to {output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    default_tokenizer_path, default_input_file_path = default_paths()
    parser = argparse.ArgumentParser(
        description="Tokenize a text dataset using a saved Hugging Face tokenizer."
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=default_tokenizer_path,
        help=f"Tokenizer JSON path (default: {default_tokenizer_path})",
    )
    parser.add_argument(
        "--input-file-path",
        type=Path,
        default=default_input_file_path,
        help=f"Input text path (default: {default_input_file_path})",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Output path (default: <input-dir>/<input-stem>/<input-stem>_tokenized)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    tokenize_dataset(args.tokenizer_path, args.input_file_path, args.output_path)


if __name__ == "__main__":
    main()
