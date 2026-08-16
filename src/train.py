# Copyright Pathway Technology, Inc.

import argparse
import hashlib
import json
import os
import re
import time
import tomllib
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__:
    from . import bdh
else:
    import bdh
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# On a Mac you can also try
# device=torch.device('mps')

dtype = (
    "bfloat16"
    if torch.cuda.is_available()
    and torch.cuda.is_bf16_supported(including_emulation=False)
    else "float16"
)  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    torch.amp.autocast(device_type=device.type, dtype=ptdtype)
    if "cuda" in device.type
    else nullcontext()
)
scaler = torch.amp.GradScaler(device=device.type, enabled=(dtype == "float16"))
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
print(f"Using device: {device} with dtype {dtype}")


# Configuration
project_root = Path(__file__).resolve().parent.parent
base_config_path = project_root / "config.toml"
base_config_toml = base_config_path.read_text(encoding="utf-8")
with base_config_path.open("rb") as config_file:
    CONFIG = tomllib.load(config_file)

TRAIN_CONFIG = CONFIG["train"]
DATA_CONFIG = CONFIG.get("data", {})
MODEL_CONFIG = CONFIG.get("model", {})

BLOCK_SIZE: int = TRAIN_CONFIG["BLOCK_SIZE"]
BATCH_SIZE: int = TRAIN_CONFIG["BATCH_SIZE"]
MAX_ITERS: int = TRAIN_CONFIG["MAX_ITERS"]
LEARNING_RATE: float = TRAIN_CONFIG["LEARNING_RATE"]
WEIGHT_DECAY: float = TRAIN_CONFIG["WEIGHT_DECAY"]
LOG_FREQ: int = TRAIN_CONFIG["LOG_FREQ"]
TOKENIZER_ENABLED: bool = TRAIN_CONFIG.get("tokenizer", False)
START_FROM_CHECKPOINT: bool = TRAIN_CONFIG.get(
    "start_from_checkpoint", TRAIN_CONFIG.get("from_checkpoint", False)
)
CHECKPOINT_PATH: str = TRAIN_CONFIG.get("checkpoint_path", "")
CURRENT_CONFIG_TOML = base_config_toml


def configured_path(path: str) -> Path:
    """Resolve a configured path relative to the project root."""
    configured = Path(path)
    return configured if configured.is_absolute() else project_root / configured


def has_supported_checkpoint_format(checkpoint: Mapping[str, object]) -> bool:
    """Return whether a checkpoint declares the supported integer format version."""
    format_version = checkpoint.get("format_version")
    return type(format_version) is int and format_version == 1


def load_checkpoint_payload() -> tuple[Mapping[str, object], Path] | None:
    """Load and validate the configured checkpoint payload."""
    if not START_FROM_CHECKPOINT:
        return None
    if not CHECKPOINT_PATH:
        raise SystemExit(
            "start_from_checkpoint is enabled, but checkpoint_path is empty"
        )

    checkpoint_path = configured_path(CHECKPOINT_PATH)
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint file not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise SystemExit(
            f"Could not load checkpoint {checkpoint_path}: {error}"
        ) from error

    if not isinstance(checkpoint, Mapping) or not has_supported_checkpoint_format(
        checkpoint
    ):
        print(
            f"Unsupported checkpoint format for {checkpoint_path}; "
            "falling back to config.toml"
        )
        return None

    return checkpoint, checkpoint_path


def restore_checkpoint_files(
    checkpoint: Mapping[str, object],
) -> tuple[Path, Path | None]:
    """Persist checkpoint config/tokenizer artifacts under unique filenames."""
    if not has_supported_checkpoint_format(checkpoint):
        raise SystemExit("Unsupported checkpoint format: expected format_version = 1")

    config_toml = checkpoint.get("config_toml")
    tokenizer_json = checkpoint.get("tokenizer_json")
    if not isinstance(config_toml, str):
        raise SystemExit("Checkpoint format 1 is missing a string config_toml")
    if tokenizer_json is not None and not isinstance(tokenizer_json, str):
        raise SystemExit(
            "Checkpoint format 1 contains a tokenizer_json value that is not a string"
        )

    artifact_hash = hashlib.sha256(
        config_toml.encode("utf-8")
        + b"\0"
        + (tokenizer_json or "").encode("utf-8")
    ).hexdigest()[:8]

    config_output_dir = project_root / "configs" / "restored"
    config_output_dir.mkdir(parents=True, exist_ok=True)
    restored_config_path = config_output_dir / f"restored_config-{artifact_hash}.toml"
    restored_config_path.write_text(config_toml, encoding="utf-8")

    restored_tokenizer_path = None
    if tokenizer_json is not None:
        tokenizer_output_dir = project_root / "tokenizers" / "restored"
        tokenizer_output_dir.mkdir(parents=True, exist_ok=True)
        restored_tokenizer_path = (
            tokenizer_output_dir / f"restored_tokenizer-{artifact_hash}.json"
        )
        restored_tokenizer_path.write_text(tokenizer_json, encoding="utf-8")

    return restored_config_path, restored_tokenizer_path


def replace_toml_table(
    base_toml: str,
    replacement_toml: str,
    table_name: str,
) -> str:
    """Replace one top-level TOML table while preserving the other tables."""
    table_header = f"[{table_name}]"

    def table_bounds(toml_text: str) -> tuple[list[str], int, int]:
        lines = toml_text.splitlines(keepends=True)
        start = next(
            (index for index, line in enumerate(lines) if line.strip() == table_header),
            None,
        )
        if start is None:
            raise ValueError(f"Could not find TOML table {table_header}")

        end = next(
            (
                index
                for index in range(start + 1, len(lines))
                if re.match(r"^\s*\[[^\[].*\]\s*$", lines[index])
            ),
            len(lines),
        )
        return lines, start, end

    base_lines, base_start, base_end = table_bounds(base_toml)
    replacement_lines, _, _ = table_bounds(replacement_toml)
    replacement_start = next(
        index
        for index, line in enumerate(replacement_lines)
        if line.strip() == table_header
    )
    replacement_end = next(
        (
            index
            for index in range(replacement_start + 1, len(replacement_lines))
            if re.match(r"^\s*\[[^\[].*\]\s*$", replacement_lines[index])
        ),
        len(replacement_lines),
    )
    replacement_section = replacement_lines[replacement_start:replacement_end]
    if replacement_section and not replacement_section[-1].endswith(("\n", "\r")):
        replacement_section.append("\n")
    return "".join(
        base_lines[:base_start]
        + replacement_section
        + base_lines[base_end:]
    )


def configure_training_context() -> tuple[Mapping[str, object], Path] | None:
    """Apply checkpoint model artifacts while retaining base training settings."""
    global CURRENT_CONFIG_TOML, MODEL_CONFIG, BDH_CONFIG, tokenizer, tokenizer_path

    checkpoint_info = load_checkpoint_payload()
    if checkpoint_info is not None:
        checkpoint, _ = checkpoint_info
        config_toml = checkpoint["config_toml"]
        assert isinstance(config_toml, str)
        restored_config_path, restored_tokenizer_path = restore_checkpoint_files(
            checkpoint
        )
        try:
            restored_config = tomllib.loads(config_toml)
        except tomllib.TOMLDecodeError as error:
            raise SystemExit(
                f"Could not parse restored checkpoint config {restored_config_path}: {error}"
            ) from error

        model_config = restored_config.get("model")
        if not isinstance(model_config, dict):
            raise SystemExit(
                f"Restored checkpoint config {restored_config_path} has no [model] table"
            )

        MODEL_CONFIG = model_config
        CURRENT_CONFIG_TOML = replace_toml_table(
            base_config_toml,
            config_toml,
            "model",
        )
        tokenizer_path = restored_tokenizer_path
    else:
        MODEL_CONFIG = CONFIG.get("model", {})
        CURRENT_CONFIG_TOML = base_config_toml
        tokenizer_path = configured_path(
            TRAIN_CONFIG.get(
                "tokenizer_path",
                CONFIG.get("tokenizer", {}).get(
                    "OUTPUT_PATH", "tokenizers/tokenizer.json"
                ),
            )
        )

    should_load_tokenizer = tokenizer_path is not None and (
        TOKENIZER_ENABLED or checkpoint_info is not None
    )
    if TOKENIZER_ENABLED and tokenizer_path is None:
        raise SystemExit(
            "Tokenizer training is enabled, but the checkpoint has no tokenizer_json"
        )

    if should_load_tokenizer:
        try:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
        except Exception as error:
            raise SystemExit(
                f"Could not load tokenizer {tokenizer_path}: {error}"
            ) from error
    else:
        tokenizer = None

    try:
        BDH_CONFIG = bdh.BDHConfig(**MODEL_CONFIG)
    except (TypeError, ValueError) as error:
        source = "restored checkpoint" if checkpoint_info is not None else "config.toml"
        raise SystemExit(f"Invalid model configuration in {source}: {error}") from error

    if tokenizer is not None:
        tokenizer_vocab_size = tokenizer.get_vocab_size()
        if checkpoint_info is not None:
            if BDH_CONFIG.vocab_size != tokenizer_vocab_size:
                raise SystemExit(
                    "Restored model vocab_size does not match the restored tokenizer: "
                    f"{BDH_CONFIG.vocab_size} != {tokenizer_vocab_size}"
                )
        else:
            BDH_CONFIG.vocab_size = tokenizer_vocab_size

    return checkpoint_info


def load_checkpoint(
    model: nn.Module,
    checkpoint_info: tuple[Mapping[str, object], Path] | None = None,
) -> None:
    """Load the validated checkpoint weights into the model."""
    if not START_FROM_CHECKPOINT:
        return
    if checkpoint_info is None:
        checkpoint_info = load_checkpoint_payload()
    if checkpoint_info is None:
        return
    checkpoint, checkpoint_path = checkpoint_info

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise SystemExit(
            f"Checkpoint {checkpoint_path} is missing model_state_dict"
        )
    try:
        model.load_state_dict(state_dict)
    except Exception as error:
        raise SystemExit(
            f"Could not load checkpoint weights {checkpoint_path}: {error}"
        ) from error

    print(f"Loaded checkpoint: {checkpoint_path}")

input_file_path = configured_path(DATA_CONFIG.get("INPUT_FILE_PATH", "input.txt"))
tokenizer_path: Path | None = None
tokenizer: Tokenizer | None = None
BDH_CONFIG = bdh.BDHConfig(**MODEL_CONFIG)


def tokenized_data_path(path: Path) -> Path:
    """Return the generated token-ID file path for an input text file."""
    input_stem = path.stem
    return path.parent / input_stem / f"{input_stem}_tokenized"


def encode(text: str) -> list[int]:
    """Encode evaluation text using the training representation."""
    if tokenizer is not None:
        return tokenizer.encode(text).ids
    return list(text.encode("utf-8"))


def decode(token_ids: Sequence[int] | torch.Tensor) -> str:
    """Decode generated token IDs using the training representation."""
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.detach().to("cpu").flatten().tolist()
    else:
        token_ids = list(token_ids)

    if tokenizer is not None:
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    return bytes(token_ids).decode("utf-8", errors="backslashreplace")


def format_duration(duration: float) -> str:
    """Format a duration for progress output."""
    duration = max(0, int(round(duration)))
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02}m {seconds:02}s"
    if minutes:
        return f"{minutes}m {seconds:02}s"
    return f"{seconds}s"


# Fetch the tiny Shakespeare dataset
def fetch_data():
    print(f"Using data file: {input_file_path}")
    if not os.path.exists(input_file_path):
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(input_file_path, "w") as f:
            f.write(requests.get(data_url).text)


def get_batch(split):
    data_path = (
        tokenized_data_path(input_file_path)
        if TOKENIZER_ENABLED
        else input_file_path
    )
    if not data_path.is_file():
        if TOKENIZER_ENABLED:
            raise FileNotFoundError(
                f"Tokenized dataset not found: {data_path}. "
                "Run `uv run python main.py --tokenize-dataset` first."
            )
        raise FileNotFoundError(f"Input dataset not found: {data_path}")

    data_dtype = np.uint32 if TOKENIZER_ENABLED else np.uint8
    data = np.memmap(data_path, dtype=data_dtype, mode="r")
    if split == "train":
        data = data[: int(0.9 * len(data))]
    else:
        data = data[int(0.9 * len(data)) :]
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + BLOCK_SIZE]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + BLOCK_SIZE]).astype(np.int64))
            for i in ix
        ]
    )
    if torch.cuda.is_available():
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = (
            x.pin_memory().to(device, non_blocking=True),
            y.pin_memory().to(device, non_blocking=True),
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def eval(model):
    model.eval()


def save_model(
    model,
    timestamp: int,
    config_hash: str,
    current_step: int,
    loss: float,
) -> Path:
    """Save the trained model and return the checkpoint path."""
    model_output_dir = Path(CONFIG.get("output", {}).get("MODEL_PATH", "models/"))
    if not model_output_dir.is_absolute():
        model_output_dir = project_root / model_output_dir
    model_output_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_output_dir / f"bdh-{timestamp}-{config_hash[:8]}.pth"
    model_to_save = getattr(model, "_orig_mod", model)
    checkpoint = {
        "format_version": 1,
        "model_state_dict": model_to_save.state_dict(),
        "config_toml": CURRENT_CONFIG_TOML,
        "tokenizer_json": (
            tokenizer_path.read_text(encoding="utf-8")
            if tokenizer is not None
            else None
        ),
        "step": current_step,
        "loss": float(loss),
        "tokens_trained": current_step * BATCH_SIZE * BLOCK_SIZE,
    }
    torch.save(checkpoint, model_path)
    return model_path


def save_loss_graph(
    loss_values: list[float], timestamp: int, config_hash: str
) -> Path:
    """Save a loss-versus-iteration graph and return the image path."""
    graph_output_dir = Path(CONFIG.get("output", {}).get("GRAPH_PATH", "analysis/"))
    if not graph_output_dir.is_absolute():
        graph_output_dir = project_root / graph_output_dir
    graph_output_dir.mkdir(parents=True, exist_ok=True)

    graph_path = graph_output_dir / f"bdh-loss-graph-{timestamp}-{config_hash[:8]}.png"
    iterations = range(1, len(loss_values) + 1)
    figure, axis = plt.subplots()
    axis.plot(iterations, loss_values)
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Loss")
    axis.set_title("BDH Training Loss")
    figure.tight_layout()
    figure.savefig(graph_path)
    plt.close(figure)
    return graph_path


def main(dry: bool = False):
    checkpoint_info = configure_training_context()
    fetch_data()

    model = bdh.BDH(BDH_CONFIG).to(device)
    if checkpoint_info is not None:
        load_checkpoint(model, checkpoint_info)
    params = sum([p.numel() for p in model.parameters()])
    print(f"Total parameters: {params / 1e6:.2f} million")

    training_tokens = BLOCK_SIZE * BATCH_SIZE * MAX_ITERS
    print(f"Training using {training_tokens / 1e6} million tokens")
    if dry:
        return

    x, y = get_batch("train")
    if checkpoint_info is not None:
        model.eval()
        with torch.no_grad(), ctx:
            _, checkpoint_loss = model(x, y)
        print(f"Checkpoint loss before training: {checkpoint_loss.item():.3f}")
        model.train()

    model = torch.compile(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    loss_acc = 0
    loss_steps = 0
    loss_values: list[float] = []
    current_step = 0
    current_loss = 0.0
    training_start = time.perf_counter()
    for step in range(MAX_ITERS):
        with ctx:
            logits, loss = model(x, y)
        loss_values.append(loss.item())
        current_step = step + 1
        current_loss = loss.item()
        x, y = get_batch("train")
        loss_acc += loss
        loss_steps += 1
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if (step + 1) % LOG_FREQ == 0:
            training_duration = time.perf_counter() - training_start
            completed_steps = step + 1
            estimated_total_duration = training_duration / completed_steps * MAX_ITERS
            estimated_remaining_duration = max(
                0, estimated_total_duration - training_duration
            )
            print(
                f"Step: {completed_steps}/{MAX_ITERS} "
                f"loss {loss_acc.item() / loss_steps:.3} "
                f"elapsed {format_duration(training_duration)} "
                f"ETA {format_duration(estimated_remaining_duration)}"
            )
            loss_acc = 0
            loss_steps = 0
    training_duration = time.perf_counter() - training_start
    print(f"Trained model using {training_tokens / 1e6} million tokens.")
    print(f"Training done in {training_duration:.3f}s, now generating a sample")

    evaluation_start = time.perf_counter()
    model.eval()
    prompt = torch.tensor(
        encode("To be or "), dtype=torch.long, device=device
    ).unsqueeze(0)
    ret = model.generate(prompt, max_new_tokens=100, top_k=3)
    ret_decoded = decode(ret)
    print(ret_decoded)
    evaluation_duration = time.perf_counter() - evaluation_start
    print(f"Final evaluation done in {evaluation_duration:.3f}s")

    timestamp = int(time.time())
    relevant_config = {
        "block_size": BLOCK_SIZE,
        "batch_size": BATCH_SIZE,
        "max_iters": MAX_ITERS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "input_file_path": str(input_file_path),
        "tokenized_data_path": str(tokenized_data_path(input_file_path))
        if TOKENIZER_ENABLED
        else None,
        "tokenizer": TOKENIZER_ENABLED,
        "tokenizer_path": str(tokenizer_path) if tokenizer is not None else None,
        "n_layer": BDH_CONFIG.n_layer,
        "n_embd": BDH_CONFIG.n_embd,
        "dropout": BDH_CONFIG.dropout,
        "n_head": BDH_CONFIG.n_head,
        "mlp_internal_dim_multiplier": BDH_CONFIG.mlp_internal_dim_multiplier,
        "vocab_size": BDH_CONFIG.vocab_size,
    }
    config_payload = json.dumps(
        relevant_config, sort_keys=True, separators=(",", ":")
    )
    config_hash = hashlib.sha256(config_payload.encode("utf-8")).hexdigest()

    print(f"Timestamp: {timestamp}")
    print(f"Config hash: {config_hash}")

    model_path = save_model(
        model,
        timestamp,
        config_hash,
        current_step=current_step,
        loss=current_loss,
    )
    print(f"Model saved to: {model_path}")

    graph_path = save_loss_graph(loss_values, timestamp, config_hash)
    print(f"Loss graph saved to: {graph_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BDH.")
    parser.add_argument(
        "--dry",
        action="store_true",
        help="check BDH setup and report model size without training",
    )
    main(dry=parser.parse_args().dry)
