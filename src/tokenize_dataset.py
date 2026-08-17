"""Tokenize a text dataset and save its token IDs as a binary array."""

from __future__ import annotations

import argparse
import json
import time
import tomllib
from collections.abc import Sequence
from pathlib import Path

import gigatoken as gt
import numpy as np
from tokenizers.pre_tokenizers import ByteLevel

from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.toml"


def resolve_config_path(config_path: str | Path | None) -> Path:
    """Resolve a config path, defaulting to the project's config.toml."""
    if config_path is None:
        return DEFAULT_CONFIG_PATH
    path = Path(config_path)
    return path if path.is_absolute() else Path.cwd() / path


def configured_path(path: str) -> Path:
    """Resolve a configured path relative to the project root."""
    configured = Path(path)
    return configured if configured.is_absolute() else PROJECT_ROOT / configured


def default_paths(config_path: str | Path | None = None) -> tuple[Path, Path]:
    """Read the tokenizer and dataset paths from the project configuration."""
    with resolve_config_path(config_path).open("rb") as config_file:
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


def load_gigatokenizer(tokenizer_path: Path) -> gt.Tokenizer:
    """Load a Hugging Face tokenizer into gigatoken's native backend.

    Older byte-level BPE files can omit byte-alphabet entries that never
    occurred in their training data. gigatoken requires the complete byte
    alphabet, so add those entries to an in-memory copy of the tokenizer
    configuration without changing the on-disk tokenizer.
    """
    hf_tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer_config = json.loads(hf_tokenizer.to_str())
    vocab = tokenizer_config["model"]["vocab"]
    missing_alphabet = sorted(set(ByteLevel.alphabet()) - set(vocab))

    if missing_alphabet:
        next_id = max(vocab.values()) + 1
        for token in missing_alphabet:
            vocab[token] = next_id
            next_id += 1
        hf_tokenizer = Tokenizer.from_str(json.dumps(tokenizer_config))

    return gt.Tokenizer(hf_tokenizer)


def tokenize_dataset(
    tokenizer_path: Path, input_file_path: Path, output_path: Path | None = None
) -> Path:
    """Tokenize ``input_file_path`` and save raw uint32 token IDs."""
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    if not input_file_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file_path}")

    start_time = time.perf_counter()
    output_path = output_path or output_path_for(input_file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = load_gigatokenizer(tokenizer_path)
    text = input_file_path.read_text(encoding="utf-8")
    encoded = tokenizer.encode_batch([text])[0]
    token_ids = np.asarray(encoded.tolist(), dtype=np.uint32)
    token_ids.tofile(output_path)

    print(f"Tokenized {input_file_path} into {len(token_ids):,} tokens")
    print(f"Tokenized data saved to {output_path}")
    print(f"Tokenized dataset in {time.perf_counter() - start_time:.2f} seconds")
    return output_path


def build_parser(
    config_path: str | Path | None = None,
) -> argparse.ArgumentParser:
    config_path = resolve_config_path(config_path)
    default_tokenizer_path, default_input_file_path = default_paths(config_path)
    parser = argparse.ArgumentParser(
        description="Tokenize a text dataset using gigatoken and a saved tokenizer."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=config_path,
        help="configuration TOML file (default: config.toml)",
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
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    config_args, _ = config_parser.parse_known_args(argv)
    args = build_parser(config_args.config).parse_args(argv)
    tokenize_dataset(args.tokenizer_path, args.input_file_path, args.output_path)


if __name__ == "__main__":
    main()
