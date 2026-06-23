"""Run WebNav-RL tasks with a MiniMind checkpoint generator."""

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


TOOL_BLOCK_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


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


def clean_completion(text: str) -> str:
    text = text.strip()
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>", 1)[0].strip()
    match = TOOL_BLOCK_RE.search(text)
    if match is not None:
        return match.group(0).strip()
    return text.strip()


class MiniMindWebNavGenerator:
    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_path: str,
        device_name: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
    ) -> None:
        self.device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name)
        self.tokenizer = load_tokenizer(tokenizer_path, allow_byte_fallback=False)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model = MiniMindForCausalLM(MiniMindConfig.from_dict(checkpoint["model_config"])).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    def __call__(self, messages: list[dict[str, str]]) -> str:
        prompt = render_prompt(messages)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        generated_ids = generate(
            model=self.model,
            input_ids=input_ids,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(generated_ids[0].tolist())
        return clean_completion(text[len(prompt) :])


def summarize(trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(trajectories)
    success = sum(1 for row in trajectories if row["summary"].get("success"))
    invalid = sum(int(row["summary"].get("invalid_tool_calls", 0)) for row in trajectories)
    format_errors = sum(int(row["summary"].get("format_errors", 0)) for row in trajectories)
    submitted = sum(1 for row in trajectories if row["summary"].get("termination") == "submitted")
    steps = [int(row["summary"].get("model_steps", 0)) for row in trajectories]
    by_template: dict[str, dict[str, int]] = {}
    for row in trajectories:
        template = row.get("template") or row["summary"].get("template") or "unknown"
        bucket = by_template.setdefault(template, {"total": 0, "success": 0})
        bucket["total"] += 1
        if row["summary"].get("success"):
            bucket["success"] += 1
    return {
        "total": total,
        "success": success,
        "success_rate": round(success / max(1, total), 4),
        "submitted": submitted,
        "submitted_rate": round(submitted / max(1, total), 4),
        "invalid_tool_calls": invalid,
        "format_errors": format_errors,
        "avg_steps": round(sum(steps) / max(1, total), 3),
        "by_template": by_template,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Evaluate a MiniMind checkpoint in WebNav-RL.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--webnav-root", default="D:/job/Program/WebNav-RL")
    parser.add_argument("--tasks", default="D:/job/Program/WebNav-RL/tasks/eval_tasks.jsonl")
    parser.add_argument("--tokenizer-path", default="outputs/tooluse_init/tokenizer")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    webnav_root = Path(args.webnav_root)
    if str(webnav_root) not in sys.path:
        sys.path.insert(0, str(webnav_root))

    from rollout.model_runner import run_model_task

    tasks = load_jsonl(Path(args.tasks), limit=args.limit)
    generator = MiniMindWebNavGenerator(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer_path,
        device_name=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    trajectories = []
    for index, task in enumerate(tasks, start=1):
        result = run_model_task(task, generator, max_steps=args.max_steps)
        result["template"] = task.get("template")
        result["difficulty"] = task.get("difficulty")
        result["page_type"] = task.get("page_type")
        trajectories.append(result)
        print(
            f"{index:04d}/{len(tasks):04d} {task['task_id']} "
            f"success={result['summary'].get('success')} "
            f"invalid={result['summary'].get('invalid_tool_calls')} "
            f"steps={result['summary'].get('model_steps')}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in trajectories:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "checkpoint": args.checkpoint,
        "tasks": args.tasks,
        "limit": args.limit,
        "max_steps": args.max_steps,
        "summary": summarize(trajectories),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
