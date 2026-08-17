import tomllib
from pathlib import Path

from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

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
    """Resolve a config path relative to the project root."""
    configured = Path(path)
    return configured if configured.is_absolute() else PROJECT_ROOT / configured


def main(config_path: str | Path | None = None) -> None:
    config_path = resolve_config_path(config_path)
    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)

    tokenizer_config = config["tokenizer"]
    input_text = configured_path(tokenizer_config["INPUT_TEXT"])
    output_path = configured_path(tokenizer_config["OUTPUT_PATH"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=tokenizer_config["VOCAB_SIZE"],
        special_tokens=tokenizer_config["SPECIAL_TOKENS"],
    )

    print(f"Training tokenizer on {input_text}")
    tokenizer.train([str(input_text)], trainer)
    tokenizer.save(str(output_path))
    print(f"Tokenizer saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a tokenizer.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="configuration TOML file (default: config.toml)",
    )
    main(parser.parse_args().config)
