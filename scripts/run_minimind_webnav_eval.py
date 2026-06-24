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


def compact_messages(
    messages: list[dict[str, str]],
    include_latest_action: bool = True,
) -> list[dict[str, str]]:
    """Keep the task instruction and the most recent action/observation pair."""
    first_assistant = next(
        (index for index, message in enumerate(messages) if message["role"] == "assistant"),
        None,
    )
    if first_assistant is None:
        return list(messages)

    prefix = list(messages[:first_assistant])
    last_tool = next(
        (index for index in range(len(messages) - 1, first_assistant - 1, -1) if messages[index]["role"] == "tool"),
        None,
    )
    if last_tool is None:
        return prefix + list(messages[first_assistant:])

    tail_start = last_tool
    if include_latest_action and messages[last_tool - 1]["role"] == "assistant":
        tail_start = last_tool - 1
    return prefix + list(messages[tail_start:])


def prepare_prompt(
    messages: list[dict[str, str]],
    tokenizer: Any,
    max_prompt_tokens: int,
) -> tuple[str, list[int], bool]:
    prompt = render_prompt(messages)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_ids) <= max_prompt_tokens:
        return prompt, prompt_ids, False

    compacted_prompt = render_prompt(compact_messages(messages))
    compacted_ids = tokenizer.encode(compacted_prompt, add_special_tokens=False)
    if len(compacted_ids) <= max_prompt_tokens:
        return compacted_prompt, compacted_ids, True

    observation_prompt = render_prompt(compact_messages(messages, include_latest_action=False))
    observation_ids = tokenizer.encode(observation_prompt, add_special_tokens=False)
    if len(observation_ids) <= max_prompt_tokens:
        return observation_prompt, observation_ids, True

    raise ValueError(
        f"Prompt remains too long after message compaction: {len(observation_ids)} > {max_prompt_tokens} tokens."
    )


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
        self.max_prompt_tokens = self.model.config.max_position_embeddings
        self.prompt_count = 0
        self.compacted_prompt_count = 0
        self.max_prompt_length = 0

    def __call__(self, messages: list[dict[str, str]]) -> str:
        _, prompt_ids, compacted = prepare_prompt(messages, self.tokenizer, self.max_prompt_tokens)
        self.prompt_count += 1
        self.compacted_prompt_count += int(compacted)
        self.max_prompt_length = max(self.max_prompt_length, len(prompt_ids))
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
        completion_ids = generated_ids[0, input_ids.shape[1] :].tolist()
        return clean_completion(self.tokenizer.decode(completion_ids))

    def prompt_stats(self) -> dict[str, int]:
        return {
            "prompt_count": self.prompt_count,
            "compacted_prompt_count": self.compacted_prompt_count,
            "max_prompt_tokens": self.max_prompt_length,
            "prompt_token_budget": self.max_prompt_tokens,
        }


class OracleFirstOpenGenerator:
    """Force the task-provided start page once, then delegate to the model."""

    def __init__(self, delegate: Any, start_page: str) -> None:
        self.delegate = delegate
        self.start_page = start_page
        self.used = False

    def __call__(self, messages: list[dict[str, str]]) -> str:
        if not self.used:
            self.used = True
            payload = {"name": "open_page", "arguments": {"page_id": self.start_page}}
            return f"<tool_call>{json.dumps(payload, ensure_ascii=False)}</tool_call>"
        return self.delegate(messages)


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
    parser.add_argument("--metadata", default=None, help="Optional WebNav page metadata for V2/custom tasks.")
    parser.add_argument("--tokenizer-path", default="outputs/tooluse_init/tokenizer")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument(
        "--oracle-first-open",
        action="store_true",
        help="Ablation: force open_page(task.start_page) once, then use the model.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    webnav_root = Path(args.webnav_root)
    if str(webnav_root) not in sys.path:
        sys.path.insert(0, str(webnav_root))

    from env.browser_env import BrowserEnv
    from env.page_state import PageStore
    from rollout.model_runner import run_model_task

    tasks = load_jsonl(Path(args.tasks), limit=args.limit)
    page_store = PageStore(args.metadata) if args.metadata is not None else None
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
        env = BrowserEnv(page_store=page_store) if page_store is not None else None
        task_generator = (
            OracleFirstOpenGenerator(generator, task["start_page"])
            if args.oracle_first_open
            else generator
        )
        result = run_model_task(task, task_generator, max_steps=args.max_steps, env=env)
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
        "metadata": args.metadata,
        "limit": args.limit,
        "max_steps": args.max_steps,
        "oracle_first_open": args.oracle_first_open,
        "prompt_stats": generator.prompt_stats(),
        "summary": summarize(trajectories),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
