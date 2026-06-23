"""Evaluate MiniMind SFT next-action generation format on converted examples."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from scripts.generate import generate
from src.data import load_tokenizer
from src.model import MiniMindConfig, MiniMindForCausalLM


TOOL_CALL_RE = re.compile(r"^\s*<tool_call>(.*?)</tool_call>(?:<\|im_end\|>)?\s*$", re.DOTALL)
VALID_TOOLS = {"open_page", "click", "get_visible_text", "submit_answer"}


def load_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def parse_completion(completion: str) -> tuple[dict[str, Any] | None, str | None]:
    match = TOOL_CALL_RE.match(completion)
    if match is None:
        return None, "missing_tool_call_wrapper"
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None, "invalid_json"
    if sorted(payload) != ["arguments", "name"]:
        return payload, "invalid_keys"
    if not isinstance(payload["name"], str) or not isinstance(payload["arguments"], dict):
        return payload, "invalid_types"
    return payload, None


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Evaluate MiniMind SFT tool-call format.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="outputs/minimind_sft/sft_eval_next_action.jsonl")
    parser.add_argument("--tokenizer-path", default="outputs/tooluse_init/tokenizer")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", default="reports/minimind_sft_format_eval.json")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    rows = load_rows(Path(args.data), limit=args.limit)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tokenizer = load_tokenizer(args.tokenizer_path, allow_byte_fallback=False)
    model = MiniMindForCausalLM(MiniMindConfig.from_dict(checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    records = []
    counts: dict[str, int] = {
        "total": 0,
        "wrapper_ok": 0,
        "json_ok": 0,
        "valid_tool_name": 0,
        "tool_name_match": 0,
        "arguments_exact_match": 0,
        "target_exact_match": 0,
    }
    error_counts: dict[str, int] = {}

    for row in rows:
        prompt_ids = tokenizer.encode(row["prompt_text"], add_special_tokens=False)
        generated = generate(
            model=model,
            input_ids=torch.tensor([prompt_ids], dtype=torch.long, device=device),
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            top_k=None,
            top_p=None,
            eos_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(generated[0].tolist())
        completion = text[len(row["prompt_text"]) :]
        payload, error = parse_completion(completion)
        target_payload, _ = parse_completion(row["target_text"])

        counts["total"] += 1
        if completion.strip().startswith("<tool_call>") and "</tool_call>" in completion:
            counts["wrapper_ok"] += 1
        if payload is not None and error is None:
            counts["json_ok"] += 1
            if payload["name"] in VALID_TOOLS:
                counts["valid_tool_name"] += 1
            if target_payload is not None and payload["name"] == target_payload["name"]:
                counts["tool_name_match"] += 1
            if target_payload is not None and payload["arguments"] == target_payload["arguments"]:
                counts["arguments_exact_match"] += 1
        else:
            error_counts[error or "unknown"] = error_counts.get(error or "unknown", 0) + 1
        if completion.strip() == row["target_text"].strip():
            counts["target_exact_match"] += 1

        records.append(
            {
                "id": row["id"],
                "target_tool": row["tool_name"],
                "completion": completion,
                "target_text": row["target_text"],
                "parsed": payload,
                "error": error,
            }
        )

    total = max(1, counts["total"])
    metrics = {key: round(value / total, 4) for key, value in counts.items() if key != "total"}
    result = {
        "checkpoint": args.checkpoint,
        "data": args.data,
        "limit": args.limit,
        "counts": counts,
        "metrics": metrics,
        "error_counts": error_counts,
        "sample_records": records[:10],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
