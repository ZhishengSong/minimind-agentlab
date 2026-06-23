"""Inspect MiniMind next-action SFT JSONL records."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_tokenizer


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Inspect converted MiniMind SFT data.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--tokenizer-path", default="outputs/tooluse_init/tokenizer")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--show", type=int, default=2)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input), limit=args.limit)
    tokenizer = load_tokenizer(args.tokenizer_path, allow_byte_fallback=False)
    lengths = []
    target_lengths = []
    tool_counts: dict[str, int] = {}
    for row in rows:
        lengths.append(len(tokenizer.encode(row["text"], add_special_tokens=False)))
        target_lengths.append(len(tokenizer.encode(row["target_text"], add_special_tokens=False)))
        tool_counts[row["tool_name"]] = tool_counts.get(row["tool_name"], 0) + 1

    report = {
        "input": args.input,
        "num_examples": len(rows),
        "tokenizer_path": args.tokenizer_path,
        "tool_counts": tool_counts,
        "total_tokens": {
            "min": min(lengths) if lengths else 0,
            "mean": round(statistics.mean(lengths), 2) if lengths else 0,
            "max": max(lengths) if lengths else 0,
        },
        "target_tokens": {
            "min": min(target_lengths) if target_lengths else 0,
            "mean": round(statistics.mean(target_lengths), 2) if target_lengths else 0,
            "max": max(target_lengths) if target_lengths else 0,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    for row in rows[: args.show]:
        print("\n--- example", row["id"], "---")
        print(row["text"][:2000])


if __name__ == "__main__":
    main()
