"""Convert WebNav/WebGym tool-use chats into MiniMind next-action SFT data."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_tokenizer


SYSTEM_PROMPT = """You are a web navigation agent. Respond with exactly one tool call:
<tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>
Available tools:
- open_page(page_id)
- click(element_id)
- get_visible_text()
- submit_answer(answer)
Use only information returned by the tools."""

TOOL_CALL_RE = re.compile(r"^<tool_call>(.*)</tool_call>$", re.DOTALL)


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            if limit is not None and len(rows) >= limit:
                break
    return rows


def unwrap_row(row: dict[str, Any]) -> dict[str, Any]:
    if "trajectory" in row and isinstance(row["trajectory"], dict):
        return row["trajectory"]
    return row


def validate_tool_call(content: str) -> dict[str, Any]:
    match = TOOL_CALL_RE.match(content.strip())
    if match is None:
        raise ValueError(f"Assistant message is not a tool call: {content!r}")
    payload = json.loads(match.group(1))
    if sorted(payload) != ["arguments", "name"]:
        raise ValueError(f"Invalid tool call keys: {payload}")
    if not isinstance(payload["name"], str):
        raise TypeError(f"Tool call name must be a string: {payload}")
    if not isinstance(payload["arguments"], dict):
        raise TypeError(f"Tool call arguments must be an object: {payload}")
    return payload


def render_message(message: dict[str, str]) -> str:
    role = message["role"]
    content = message["content"]
    if role == "tool":
        return f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"
    if role in {"system", "user"}:
        return f"<|im_start|>{role}\n{content}<|im_end|>\n"
    if role == "assistant":
        return f"<|im_start|>assistant\n{content}<|im_end|>\n"
    raise ValueError(f"Unsupported role: {role!r}")


def render_prompt(messages: list[dict[str, str]]) -> str:
    return "".join(render_message(message) for message in messages) + "<|im_start|>assistant\n"


def normalize_messages(row: dict[str, Any], include_system: bool) -> list[dict[str, str]]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"Row has no messages: {row.get('id') or row.get('task_id')}")

    normalized: list[dict[str, str]] = []
    if include_system and messages[0].get("role") != "system":
        normalized.append({"role": "system", "content": SYSTEM_PROMPT})

    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported message role: {role!r}")
        if not isinstance(content, str):
            raise TypeError(f"Message content must be a string for role {role!r}")
        normalized.append({"role": role, "content": content})
    return normalized


def row_to_examples(
    source_row: dict[str, Any],
    include_system: bool,
    include_failures: bool,
) -> list[dict[str, Any]]:
    row = unwrap_row(source_row)
    summary = row.get("summary", {})
    if not include_failures and isinstance(summary, dict) and summary.get("success") is False:
        return []

    row_id = row.get("id") or row.get("task_id") or source_row.get("id") or source_row.get("task_id")
    if row_id is None:
        raise ValueError("Row is missing id/task_id")

    messages = normalize_messages(row, include_system=include_system)
    context: list[dict[str, str]] = []
    examples: list[dict[str, Any]] = []
    assistant_turn = 0

    for message_index, message in enumerate(messages):
        if message["role"] == "assistant":
            payload = validate_tool_call(message["content"])
            assistant_turn += 1
            prompt_text = render_prompt(context)
            target_text = f"{message['content']}<|im_end|>\n"
            text = prompt_text + target_text
            examples.append(
                {
                    "id": f"{row_id}::turn{assistant_turn:02d}",
                    "source_id": row_id,
                    "message_index": message_index,
                    "assistant_turn": assistant_turn,
                    "tool_name": payload["name"],
                    "arguments": payload["arguments"],
                    "prompt_text": prompt_text,
                    "target_text": target_text,
                    "text": text,
                    "prompt_chars": len(prompt_text),
                    "target_chars": len(target_text),
                    "summary": summary,
                }
            )
        context.append(message)
    return examples


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def percentile(values: list[int], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, round((pct / 100.0) * (len(sorted_values) - 1)))
    return float(sorted_values[index])


def build_report(
    examples: list[dict[str, Any]],
    skipped_for_length: int,
    tokenizer: Any | None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "num_examples": len(examples),
        "skipped_for_length": skipped_for_length,
        "tool_counts": {},
    }
    for example in examples:
        tool_name = example["tool_name"]
        report["tool_counts"][tool_name] = report["tool_counts"].get(tool_name, 0) + 1

    if tokenizer is not None and examples:
        total_lengths = [int(example["total_tokens"]) for example in examples]
        prompt_lengths = [int(example["prompt_tokens"]) for example in examples]
        target_lengths = [int(example["target_tokens"]) for example in examples]
        report["token_lengths"] = {
            "total_min": min(total_lengths),
            "total_mean": round(statistics.mean(total_lengths), 2),
            "total_p50": percentile(total_lengths, 50),
            "total_p95": percentile(total_lengths, 95),
            "total_max": max(total_lengths),
            "prompt_mean": round(statistics.mean(prompt_lengths), 2),
            "target_mean": round(statistics.mean(target_lengths), 2),
        }
    return report


def convert_file(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = None
    if args.tokenizer_path:
        tokenizer = load_tokenizer(args.tokenizer_path, allow_byte_fallback=False)

    rows = load_jsonl(Path(args.input), limit=args.limit)
    examples: list[dict[str, Any]] = []
    skipped_for_length = 0

    for row in rows:
        for example in row_to_examples(row, include_system=args.include_system, include_failures=args.include_failures):
            if tokenizer is not None:
                prompt_tokens = token_count(tokenizer, example["prompt_text"])
                target_tokens = token_count(tokenizer, example["target_text"])
                total_tokens = prompt_tokens + target_tokens
                example["prompt_tokens"] = prompt_tokens
                example["target_tokens"] = target_tokens
                example["total_tokens"] = total_tokens
                if args.max_seq_len and total_tokens > args.max_seq_len:
                    skipped_for_length += 1
                    continue
            examples.append(example)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    report = build_report(examples, skipped_for_length=skipped_for_length, tokenizer=tokenizer)
    report.update(
        {
            "input": str(args.input),
            "output": str(args.output),
            "source_rows": len(rows),
            "include_system": args.include_system,
            "include_failures": args.include_failures,
            "max_seq_len": args.max_seq_len,
            "tokenizer_path": args.tokenizer_path,
        }
    )

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Convert WebNav/WebGym SFT chats to MiniMind next-action SFT.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--tokenizer-path", default="outputs/tooluse_init/tokenizer")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-system", action="store_true", default=True)
    parser.add_argument("--no-include-system", action="store_false", dest="include_system")
    parser.add_argument("--include-failures", action="store_true", default=False)
    args = parser.parse_args()

    report = convert_file(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
