"""Evaluate causal LM loss on a fixed JSONL validation slice."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Iterator
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data import CausalLMCollator, load_tokenizer
from src.model import MiniMindConfig, MiniMindForCausalLM
from src.train.config import PretrainConfig


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def resolve_autocast_dtype(dtype_name: str) -> torch.dtype | None:
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    return None


def iter_jsonl_examples(
    data_path: Path,
    text_field: str,
    skip_examples: int,
    num_examples: int,
) -> Iterator[tuple[str, str]]:
    """Yield selected raw JSONL lines and text fields."""
    yielded = 0
    seen = 0
    with data_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw_line = line.rstrip("\n")
            if not raw_line:
                continue
            if seen < skip_examples:
                seen += 1
                continue

            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {data_path}:{line_no}") from exc
            if text_field not in record:
                raise KeyError(f"Missing field {text_field!r} in {data_path}:{line_no}")
            text = record[text_field]
            if not isinstance(text, str):
                raise TypeError(f"Field {text_field!r} must be a string in {data_path}:{line_no}")

            yield raw_line, text
            yielded += 1
            if yielded >= num_examples:
                return


def batched(examples: list[dict[str, list[int]]], batch_size: int) -> Iterator[list[dict[str, list[int]]]]:
    for start in range(0, len(examples), batch_size):
        yield examples[start : start + batch_size]


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, object]:
    train_config = PretrainConfig.from_yaml(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = MiniMindConfig.from_dict(checkpoint["model_config"])
    checkpoint_train_config = checkpoint.get("train_config", {})

    data_path = Path(args.data_path or train_config.train_data_path)
    tokenizer_path = args.tokenizer_path or checkpoint_train_config.get("tokenizer_path") or train_config.tokenizer_path
    max_seq_len = args.max_seq_len or train_config.max_seq_len
    dtype_name = args.dtype or train_config.dtype
    batch_size = args.batch_size or train_config.batch_size

    tokenizer = load_tokenizer(tokenizer_path, allow_byte_fallback=args.allow_byte_fallback)
    if tokenizer.vocab_size != model_config.vocab_size:
        print(
            "warning: tokenizer vocab size "
            f"({tokenizer.vocab_size}) != model vocab size ({model_config.vocab_size})"
        )

    selected_examples: list[dict[str, list[int]]] = []
    selected_lines: list[str] = []
    for raw_line, text in iter_jsonl_examples(
        data_path=data_path,
        text_field=args.text_field,
        skip_examples=args.skip_examples,
        num_examples=args.num_examples,
    ):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if args.add_eos and tokenizer.eos_token_id is not None:
            token_ids.append(tokenizer.eos_token_id)
        token_ids = token_ids[:max_seq_len]
        if token_ids:
            selected_examples.append({"input_ids": token_ids})
            selected_lines.append(raw_line)

    if not selected_examples:
        raise ValueError(f"No examples selected from {data_path}")

    if args.save_subset:
        subset_path = Path(args.save_subset)
        subset_path.parent.mkdir(parents=True, exist_ok=True)
        subset_path.write_text("\n".join(selected_lines) + "\n", encoding="utf-8")

    device = resolve_device(args.device)
    autocast_dtype = resolve_autocast_dtype(dtype_name)
    use_autocast = autocast_dtype is not None and device.type in {"cuda", "cpu"}

    model = MiniMindForCausalLM(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    collator = CausalLMCollator(pad_token_id=tokenizer.pad_token_id)
    total_loss = 0.0
    total_predicted_tokens = 0
    total_input_tokens = 0
    max_batch_len = 0
    start_time = time.perf_counter()

    for batch_examples in batched(selected_examples, batch_size):
        batch = collator(batch_examples)
        batch = {key: value.to(device) for key, value in batch.items()}
        max_batch_len = max(max_batch_len, int(batch["input_ids"].shape[1]))
        total_input_tokens += int(batch["attention_mask"].sum().item())

        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
            output = model(**batch)
        if output.loss is None:
            raise RuntimeError("Model did not return loss during evaluation.")

        predicted_tokens = int((batch["labels"][..., 1:] != -100).sum().item())
        total_loss += float(output.loss.detach().item()) * predicted_tokens
        total_predicted_tokens += predicted_tokens

    if total_predicted_tokens <= 0:
        raise ValueError("No predicted tokens were available for loss evaluation.")

    avg_loss = total_loss / total_predicted_tokens
    perplexity = math.exp(avg_loss) if avg_loss < 50 else float("inf")
    elapsed = time.perf_counter() - start_time

    result = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": checkpoint.get("step"),
        "config": args.config,
        "data_path": str(data_path),
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_type": type(tokenizer).__name__,
        "num_examples": len(selected_examples),
        "skip_examples": args.skip_examples,
        "batch_size": batch_size,
        "max_seq_len": max_seq_len,
        "max_batch_len": max_batch_len,
        "input_tokens": total_input_tokens,
        "predicted_tokens": total_predicted_tokens,
        "loss": round(avg_loss, 6),
        "perplexity": round(perplexity, 6) if math.isfinite(perplexity) else "inf",
        "elapsed_sec": round(elapsed, 4),
        "tokens_per_sec": round(total_input_tokens / max(elapsed, 1.0e-6), 2),
        "device": str(device),
        "dtype": dtype_name,
    }
    if args.save_subset:
        result["saved_subset"] = args.save_subset
    return result


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Evaluate fixed-slice pretraining loss.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file.")
    parser.add_argument("--config", default="configs/pretrain_minimind_local.yaml")
    parser.add_argument("--data-path", default=None, help="Validation JSONL path. Defaults to config train_data_path.")
    parser.add_argument("--tokenizer-path", default=None, help="Tokenizer path. Defaults to checkpoint/config tokenizer.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default=None, choices=["bf16", "fp16", "fp32", None])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--num-examples", type=int, default=1000)
    parser.add_argument("--skip-examples", type=int, default=0)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--save-subset", default=None, help="Optional path to save the selected raw JSONL subset.")
    parser.add_argument("--output", default=None, help="Optional path to save metrics JSON.")
    parser.add_argument("--add-eos", action="store_true", default=True)
    parser.add_argument("--no-add-eos", action="store_false", dest="add_eos")
    parser.add_argument("--allow-byte-fallback", action="store_true", default=False)
    args = parser.parse_args()

    if args.num_examples <= 0:
        raise ValueError("--num-examples must be positive.")
    if args.skip_examples < 0:
        raise ValueError("--skip-examples cannot be negative.")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.max_seq_len is not None and args.max_seq_len <= 0:
        raise ValueError("--max-seq-len must be positive.")

    result = evaluate(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"saved metrics: {output_path}")


if __name__ == "__main__":
    main()
