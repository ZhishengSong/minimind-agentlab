"""Generate from a converted MiniMind SFT prompt example."""

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


def read_example(path: Path, index: int, source_id: str | None, turn: int | None) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        for row_index, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            if source_id is not None and row.get("source_id") != source_id:
                continue
            if turn is not None and int(row.get("assistant_turn", -1)) != turn:
                continue
            if source_id is not None or turn is not None:
                return row
            if row_index == index:
                return row
    raise ValueError(f"No matching example found in {path}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Generate from one converted SFT example prompt.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="outputs/minimind_sft/sft_eval_next_action.jsonl")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--source-id", default=None)
    parser.add_argument("--turn", type=int, default=None)
    parser.add_argument("--tokenizer-path", default="outputs/tooluse_init/tokenizer")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    example = read_example(Path(args.data), args.index, args.source_id, args.turn)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tokenizer = load_tokenizer(args.tokenizer_path, allow_byte_fallback=False)
    model = MiniMindForCausalLM(MiniMindConfig.from_dict(checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prompt = str(example["prompt_text"])
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
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
    completion = text[len(prompt) :]

    result = {
        "id": example.get("id"),
        "source_id": example.get("source_id"),
        "assistant_turn": example.get("assistant_turn"),
        "tool_name": example.get("tool_name"),
        "target_text": example.get("target_text"),
        "completion": completion,
        "full_text": text,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
