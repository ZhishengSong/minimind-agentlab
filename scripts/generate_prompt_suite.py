"""Run a fixed prompt suite against one MiniMind checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from scripts.generate import generate
from src.data import load_tokenizer
from src.model import MiniMindConfig, MiniMindForCausalLM


def load_prompts(path: Path) -> list[str]:
    prompts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        prompt = line.strip()
        if prompt and not prompt.startswith("#"):
            prompts.append(prompt)
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def write_markdown(records: list[dict[str, object]], output_path: Path) -> None:
    lines = ["# Pretrain V0 Generation Suite", ""]
    for idx, record in enumerate(records, start=1):
        lines.extend(
            [
                f"## Prompt {idx}",
                "",
                "Prompt:",
                "",
                "```text",
                str(record["prompt"]),
                "```",
                "",
                "Generated:",
                "",
                "```text",
                str(record["generated"]),
                "```",
                "",
            ]
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Generate outputs for a fixed prompt suite.")
    parser.add_argument("--checkpoint", default="minimind-50k-artifacts/checkpoints/pretrain_step_050000.pt")
    parser.add_argument("--prompts", default="configs/pretrain_v0_generation_prompts.txt")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--output", default="reports/pretrain_v0_generation_suite.jsonl")
    parser.add_argument("--markdown-output", default="reports/pretrain_v0_generation_suite.md")
    parser.add_argument("--allow-byte-fallback", action="store_true", default=False)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = MiniMindConfig.from_dict(checkpoint["model_config"])
    train_config = checkpoint.get("train_config", {})
    tokenizer_path = args.tokenizer_path or train_config.get("tokenizer_path", "data/minimind/tokenizer")

    tokenizer = load_tokenizer(tokenizer_path, allow_byte_fallback=args.allow_byte_fallback)
    model = MiniMindForCausalLM(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    records: list[dict[str, object]] = []
    for prompt in load_prompts(Path(args.prompts)):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not prompt_ids:
            raise ValueError(f"Prompt produced no tokens: {prompt!r}")

        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated_ids = generate(
            model=model,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(generated_ids[0].tolist())
        record = {
            "checkpoint": args.checkpoint,
            "checkpoint_step": checkpoint.get("step"),
            "prompt": prompt,
            "generated": text,
            "device": str(device),
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
        }
        records.append(record)
        print(json.dumps(record, ensure_ascii=False))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    write_markdown(records, Path(args.markdown_output))
    print(f"saved jsonl: {output_path}")
    print(f"saved markdown: {args.markdown_output}")


if __name__ == "__main__":
    main()
