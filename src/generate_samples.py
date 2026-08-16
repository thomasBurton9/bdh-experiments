import argparse
import sys
from pathlib import Path

import torch

if __package__:
    from .inference import DEVICE, load_model, model_path_from_args
else:
    from inference import DEVICE, load_model, model_path_from_args


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ((1.0, 0.90), (0.9, 0.90), (1.0, 0.95))
DEFAULT_PROMPT = "To be or "


def generate_text(
    model,
    tokenizer,
    prompt_text: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> str:
    prompt_ids = (
        tokenizer.encode(prompt_text).ids
        if tokenizer is not None
        else list(prompt_text.encode("utf-8"))
    )
    prompt = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        generated = model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=None,
            top_p=top_p,
        )
    if tokenizer is not None:
        return tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
    return bytes(generated[0].tolist()).decode("utf-8", errors="backslashreplace")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BDH sample text.")
    parser.add_argument("--model", help="model checkpoint override")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output", default="analysis/generations.txt")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    args = parser.parse_args()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(model_path_from_args(args.model))
    with output_path.open("w", encoding="utf-8") as output_file:
        print("BDH generations", file=output_file)
        print(f"Prompt: {args.prompt!r}", file=output_file)
        print("Top-k: None (disabled)", file=output_file)
        print(f"New tokens per generation: {args.max_new_tokens}", file=output_file)

        for temperature, top_p in CONFIGS:
            print(
                f"\n=== temperature={temperature}, top_p={top_p}, top_k=None ===",
                file=output_file,
            )
            for generation_number in range(1, args.generations + 1):
                text = generate_text(
                    model,
                    tokenizer,
                    args.prompt,
                    temperature,
                    top_p,
                    args.max_new_tokens,
                )
                print(f"\n--- Generation {generation_number} ---", file=output_file)
                print(text, file=output_file)
                output_file.flush()
                print(
                    f"temperature={temperature}, top_p={top_p}: "
                    f"generation {generation_number}/{args.generations}",
                    file=sys.stderr,
                    flush=True,
                )

    print(f"Saved generations to {output_path}")


if __name__ == "__main__":
    main()
