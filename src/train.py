# Copyright Pathway Technology, Inc.

import hashlib
import json
import os
import time
import tomllib
from contextlib import nullcontext
from datetime import datetime, timezone
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# On a Mac you can also try
# device=torch.device('mps')

dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
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
BDH_CONFIG = bdh.BDHConfig()
with (project_root / "config.toml").open("rb") as config_file:
    CONFIG = tomllib.load(config_file)

TRAIN_CONFIG = CONFIG["train"]
DATA_CONFIG = CONFIG.get("data", {})

BLOCK_SIZE: int = TRAIN_CONFIG["BLOCK_SIZE"]
BATCH_SIZE: int = TRAIN_CONFIG["BATCH_SIZE"]
MAX_ITERS: int = TRAIN_CONFIG["MAX_ITERS"]
LEARNING_RATE: float = TRAIN_CONFIG["LEARNING_RATE"]
WEIGHT_DECAY: float = TRAIN_CONFIG["WEIGHT_DECAY"]
LOG_FREQ: int = TRAIN_CONFIG["LOG_FREQ"]

input_file_path = project_root / DATA_CONFIG.get("INPUT_FILE_PATH", "input.txt")


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
    # treat the file as bytes
    data = np.memmap(input_file_path, dtype=np.uint8, mode="r")
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


def save_model(model, timestamp_hash: str, config_hash: str) -> Path:
    """Save the trained model and return the checkpoint path."""
    model_output_dir = Path(CONFIG.get("output", {}).get("MODEL_PATH", "models/"))
    if not model_output_dir.is_absolute():
        model_output_dir = project_root / model_output_dir
    model_output_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_output_dir / (
        f"bdh-{timestamp_hash[:8]}-{config_hash[:8]}.pth"
    )
    model_to_save = getattr(model, "_orig_mod", model)
    torch.save(model_to_save.state_dict(), model_path)
    return model_path


def save_loss_graph(
    loss_values: list[float], timestamp_hash: str, config_hash: str
) -> Path:
    """Save a loss-versus-iteration graph and return the image path."""
    graph_output_dir = Path(CONFIG.get("output", {}).get("GRAPH_PATH", "analysis/"))
    if not graph_output_dir.is_absolute():
        graph_output_dir = project_root / graph_output_dir
    graph_output_dir.mkdir(parents=True, exist_ok=True)

    graph_path = graph_output_dir / (
        f"bdh-loss-graph-{timestamp_hash[:8]}-{config_hash[:8]}.png"
    )
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


def main():
    fetch_data()

    model = bdh.BDH(BDH_CONFIG).to(device)
    params = sum([p.numel() for p in model.parameters()])
    print(f"Total parameters: {params}")

    model = torch.compile(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    x, y = get_batch("train")

    loss_acc = 0
    loss_steps = 0
    loss_values: list[float] = []
    training_start = time.perf_counter()
    for step in range(MAX_ITERS):
        with ctx:
            logits, loss = model(x, y)
        loss_values.append(loss.item())
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
    print(f"Training done in {training_duration:.3f}s, now generating a sample")

    evaluation_start = time.perf_counter()
    model.eval()
    prompt = torch.tensor(
        bytearray("To be or ", "utf-8"), dtype=torch.long, device=device
    ).unsqueeze(0)
    ret = model.generate(prompt, max_new_tokens=100, top_k=3)
    ret_decoded = bytes(ret.to(torch.uint8).to("cpu").squeeze(0)).decode(
        errors="backslashreplace"
    )
    print(ret_decoded)
    evaluation_duration = time.perf_counter() - evaluation_start
    print(f"Final evaluation done in {evaluation_duration:.3f}s")

    timestamp = datetime.now(timezone.utc).isoformat()
    timestamp_hash = hashlib.sha256(timestamp.encode("utf-8")).hexdigest()
    relevant_config = {
        "block_size": BLOCK_SIZE,
        "batch_size": BATCH_SIZE,
        "max_iters": MAX_ITERS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "input_file_path": str(input_file_path),
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
    print(f"Timestamp hash: {timestamp_hash}")
    print(f"Config hash: {config_hash}")

    model_path = save_model(model, timestamp_hash, config_hash)
    print(f"Model saved to: {model_path}")

    graph_path = save_loss_graph(loss_values, timestamp_hash, config_hash)
    print(f"Loss graph saved to: {graph_path}")


if __name__ == "__main__":
    main()
