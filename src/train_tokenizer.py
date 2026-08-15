import tomllib
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def configured_path(path: str) -> Path:
    """Resolve a config path relative to the project root."""
    configured = Path(path)
    return configured if configured.is_absolute() else PROJECT_ROOT / configured


def main() -> None:
    with (PROJECT_ROOT / "config.toml").open("rb") as config_file:
        config = tomllib.load(config_file)

    tokenizer_config = config["tokenizer"]
    input_text = configured_path(tokenizer_config["INPUT_TEXT"])
    output_path = configured_path(tokenizer_config["OUTPUT_PATH"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel()

    trainer = BpeTrainer(
        vocab_size=tokenizer_config["VOCAB_SIZE"],
        special_tokens=tokenizer_config["SPECIAL_TOKENS"],
    )

    print(f"Training tokenizer on {input_text}")
    tokenizer.train([str(input_text)], trainer)
    tokenizer.save(str(output_path))
    print(f"Tokenizer saved to {output_path}")


if __name__ == "__main__":
    main()
