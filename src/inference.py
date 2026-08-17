# Copyright Pathway Technology, Inc.

import argparse
import tomllib
from collections.abc import Mapping
from pathlib import Path

import torch

from tokenizers import Tokenizer

if __package__:
    from . import bdh
else:
    import bdh


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = "To be or "
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configured_path(path: str) -> Path:
    configured = Path(path)
    return configured if configured.is_absolute() else PROJECT_ROOT / configured


def model_path_from_args(model_path: str | None) -> Path:
    if model_path:
        path = configured_path(model_path)
    else:
        config_path = PROJECT_ROOT / "config.toml"
        if not config_path.is_file():
            raise SystemExit(
                "No model found: provide one with --model or create config.toml"
            )
        with config_path.open("rb") as config_file:
            config = tomllib.load(config_file)
        checkpoint_path = config.get("train", {}).get("checkpoint_path", "")
        if not isinstance(checkpoint_path, str) or not checkpoint_path:
            raise SystemExit(
                "No model found: provide one with --model or set "
                "train.checkpoint_path in config.toml"
            )
        path = configured_path(checkpoint_path)

    if not path.is_file():
        raise SystemExit(f"Model file not found: {path}")
    return path


def load_model(model_path: Path):
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise SystemExit(f"Could not load model {model_path}: {error}") from error

    if (
        not isinstance(checkpoint, Mapping)
        or type(checkpoint.get("format_version")) is not int
        or checkpoint["format_version"] != 1
    ):
        raise SystemExit("Unsupported model format: expected format_version = 1")

    config_toml = checkpoint.get("config_toml")
    if not isinstance(config_toml, str):
        raise SystemExit("Model format 1 is missing config_toml")
    try:
        config = tomllib.loads(config_toml)
        model_config = config.get("model")
        train_config = config.get("train")
        if not isinstance(model_config, Mapping):
            raise TypeError("config_toml is missing a [model] table")
        if not isinstance(train_config, Mapping):
            raise TypeError("config_toml is missing a [train] table")
        block_size = train_config.get("BLOCK_SIZE")
        if type(block_size) is not int or block_size < 1:
            raise ValueError("train.BLOCK_SIZE must be a positive integer")
        model = bdh.BDH(bdh.BDHConfig(**model_config)).to(DEVICE)
    except (KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as error:
        raise SystemExit(f"Invalid model configuration: {error}") from error

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise SystemExit("Model format 1 is missing model_state_dict")
    try:
        model.load_state_dict(state_dict)
    except Exception as error:
        raise SystemExit(f"Could not load model weights: {error}") from error

    tokenizer_json = checkpoint.get("tokenizer_json")
    tokenizer = None
    if tokenizer_json is not None:
        if not isinstance(tokenizer_json, str):
            raise SystemExit("Model format 1 has an invalid tokenizer_json")
        try:
            tokenizer = Tokenizer.from_str(tokenizer_json)
        except Exception as error:
            raise SystemExit(f"Could not load model tokenizer: {error}") from error

    return model.eval(), tokenizer, block_size


def parse_top_k(value: str) -> int | None:
    if value.lower() == "none":
        return None
    try:
        top_k = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "top-k must be a positive integer or None"
        ) from error
    if top_k < 1:
        raise argparse.ArgumentTypeError("top-k must be a positive integer or None")
    return top_k


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with a BDH model.")
    parser.add_argument(
        "--model",
        help="model checkpoint path (defaults to train.checkpoint_path in config.toml)",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=f"prompt to complete (default: {DEFAULT_PROMPT!r})",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=1.0,
        help="sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--top-k",
        "--top_k",
        dest="top_k",
        type=parse_top_k,
        default=50,
        help="sample from the top k tokens, or None to disable (default: 50)",
    )
    parser.add_argument(
        "--top-p",
        "--top_p",
        dest="top_p",
        type=float,
        default=None,
        help="sample from the smallest probability mass (default: disabled)",
    )
    parser.add_argument(
        "--max-new-tokens",
        "--max_new_tokens",
        dest="max_new_tokens",
        type=int,
        default=100,
        help="number of tokens to generate (default: 100)",
    )
    args = parser.parse_args()

    model_path = model_path_from_args(args.model)
    model, tokenizer, block_size = load_model(model_path)
    prompt_ids = (
        tokenizer.encode(args.prompt).ids
        if tokenizer is not None
        else list(args.prompt.encode("utf-8"))
    )
    prompt = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16,):
        generated = model.generate(
            prompt,
            max_new_tokens=args.max_new_tokens,
            block_size=block_size,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temp,
        )

    if tokenizer is not None:
        print(tokenizer.decode(generated[0].tolist(), skip_special_tokens=True))
    else:
        print(
            bytes(generated[0].tolist()).decode(
                "utf-8", errors="backslashreplace"
            )
        )


if __name__ == "__main__":
    main()
