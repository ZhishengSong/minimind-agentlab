"""Tool-use SFT training entry point for MiniMind checkpoints."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, Dataset

from src.data import load_tokenizer
from src.model import MiniMindConfig, MiniMindForCausalLM
from src.train.logger import JsonlLogger
from src.train.optim import WarmupCosineScheduler, build_optimizer, grad_global_norm
from src.utils.config import apply_overrides, load_yaml, save_yaml
from src.utils.param_count import count_parameters, format_parameter_count


IGNORE_INDEX = -100


@dataclass(slots=True)
class SFTConfig:
    run_name: str = "sft_minimind_webnav"
    base_checkpoint: str = "outputs/tooluse_init/pretrain_step_050000_tooluse_init.pt"
    tokenizer_path: str = "outputs/tooluse_init/tokenizer"
    train_data_path: str = "outputs/minimind_sft/sft_train_next_action.jsonl"
    eval_data_path: str | None = None
    output_dir: str = "checkpoints/sft_minimind_webnav"
    log_dir: str = "logs"

    seed: int = 42
    device: str = "auto"
    dtype: str = "bf16"

    max_seq_len: int = 1024
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_steps: int = 100

    learning_rate: float = 5.0e-5
    min_learning_rate: float = 5.0e-6
    weight_decay: float = 0.01
    warmup_steps: int = 10
    max_grad_norm: float = 1.0

    log_interval: int = 10
    save_interval: int = 100
    num_workers: int = 0

    def __post_init__(self) -> None:
        if self.dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError("dtype must be one of: bf16, fp16, fp32.")
        for name in ["max_seq_len", "batch_size", "gradient_accumulation_steps", "max_steps"]:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.min_learning_rate < 0 or self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate must be in [0, learning_rate].")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative.")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive.")
        if self.log_interval <= 0 or self.save_interval <= 0:
            raise ValueError("log_interval and save_interval must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: list[str] | None = None) -> "SFTConfig":
        return cls(**apply_overrides(load_yaml(path), overrides))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NextActionSFTDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, path: str | Path, tokenizer: Any, max_seq_len: int, limit: int | None = None) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.rows = self._load_rows(limit=limit)
        if not self.rows:
            raise ValueError(f"No SFT examples loaded from {self.path}")

    def _load_rows(self, limit: int | None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if "prompt_text" not in row or "target_text" not in row:
                    raise KeyError(f"Missing prompt_text/target_text in {self.path}:{line_no}")
                rows.append(row)
                if limit is not None and len(rows) >= limit:
                    break
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        prompt_ids = self.tokenizer.encode(row["prompt_text"], add_special_tokens=False)
        target_ids = self.tokenizer.encode(row["target_text"], add_special_tokens=False)
        if not target_ids:
            raise ValueError(f"Target produced no tokens for {row.get('id')}")

        if len(target_ids) >= self.max_seq_len:
            target_ids = target_ids[-self.max_seq_len :]
            prompt_ids = []
        else:
            keep_prompt = self.max_seq_len - len(target_ids)
            prompt_ids = prompt_ids[-keep_prompt:]

        input_ids = prompt_ids + target_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + target_ids
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@dataclass(slots=True)
class SFTCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].shape[0] for item in features)
        input_ids = []
        labels = []
        attention_mask = []
        for item in features:
            pad_len = max_len - item["input_ids"].shape[0]
            input_ids.append(torch.nn.functional.pad(item["input_ids"], (0, pad_len), value=self.pad_token_id))
            labels.append(torch.nn.functional.pad(item["labels"], (0, pad_len), value=IGNORE_INDEX))
            attention_mask.append(
                torch.nn.functional.pad(torch.ones_like(item["input_ids"], dtype=torch.long), (0, pad_len), value=0)
            )
        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention_mask),
        }


def infinite_batches(dataloader: DataLoader):
    while True:
        for batch in dataloader:
            yield batch


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


def current_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024 / 1024
    return 0.0


def save_sft_checkpoint(
    output_dir: str | Path,
    step: int,
    model: MiniMindForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    config: SFTConfig,
    model_config: MiniMindConfig,
    scaler: torch.amp.GradScaler | None,
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        "train_config": config.to_dict(),
        "model_config": model_config.to_dict(),
        "base_checkpoint": config.base_checkpoint,
    }
    step_path = output_path / f"sft_step_{step:06d}.pt"
    latest_path = output_path / "latest.pt"
    torch.save(payload, step_path)
    torch.save(payload, latest_path)
    save_yaml(config.to_dict(), output_path / "sft_config.yaml")
    save_yaml(model_config.to_dict(), output_path / "model_config.yaml")
    return step_path, latest_path


def load_base_model(config: SFTConfig, device: torch.device) -> tuple[MiniMindForCausalLM, MiniMindConfig]:
    checkpoint = torch.load(config.base_checkpoint, map_location="cpu", weights_only=False)
    model_config = MiniMindConfig.from_dict(checkpoint["model_config"])
    model = MiniMindForCausalLM(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device), model_config


def train(args: argparse.Namespace, config: SFTConfig) -> None:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    tokenizer = load_tokenizer(config.tokenizer_path, allow_byte_fallback=False)
    dataset = NextActionSFTDataset(config.train_data_path, tokenizer, config.max_seq_len, limit=args.limit_examples)
    collator = SFTCollator(pad_token_id=tokenizer.pad_token_id)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=config.num_workers,
    )

    model, model_config = load_base_model(config, device)
    if tokenizer.vocab_size != model_config.vocab_size:
        raise ValueError(f"Tokenizer vocab ({tokenizer.vocab_size}) != model vocab ({model_config.vocab_size}).")
    model.train()

    optimizer = build_optimizer(model, config.learning_rate, config.weight_decay)
    scheduler = WarmupCosineScheduler(
        optimizer,
        max_steps=config.max_steps,
        warmup_steps=config.warmup_steps,
        learning_rate=config.learning_rate,
        min_learning_rate=config.min_learning_rate,
    )
    logger = JsonlLogger(config.log_dir, config.run_name)
    autocast_dtype = resolve_autocast_dtype(config.dtype)
    use_autocast = autocast_dtype is not None and device.type in {"cuda", "cpu"}
    scaler = torch.amp.GradScaler("cuda", enabled=config.dtype == "fp16" and device.type == "cuda")

    print(f"run_name: {config.run_name}")
    print(f"device/dtype: {device}/{config.dtype}")
    print(f"model params: {format_parameter_count(count_parameters(model))}")
    print(f"tokenizer: {type(tokenizer).__name__}, vocab_size={tokenizer.vocab_size}")
    print(f"examples: {len(dataset)}")
    print(f"metrics: {logger.path}")
    print(f"output_dir: {config.output_dir}")

    data_iter = infinite_batches(dataloader)
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    running_loss_count = 0
    running_target_tokens = 0
    running_examples = 0
    window_start = time.perf_counter()

    total_micro_steps = config.max_steps * config.gradient_accumulation_steps
    for micro_step in range(total_micro_steps):
        batch = next(data_iter)
        batch = {key: value.to(device) for key, value in batch.items()}
        target_tokens = int((batch["labels"] != IGNORE_INDEX).sum().item())
        if target_tokens <= 0:
            raise ValueError("Batch has no supervised target tokens.")

        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
            output = model(**batch)
            if output.loss is None:
                raise RuntimeError("Model did not return loss during SFT.")
            loss = output.loss / config.gradient_accumulation_steps

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at micro_step {micro_step}: {loss.item()}")

        scaler.scale(loss).backward()
        running_loss += float(output.loss.detach().item())
        running_loss_count += 1
        running_target_tokens += target_tokens
        running_examples += int(batch["input_ids"].shape[0])

        if (micro_step + 1) % config.gradient_accumulation_steps != 0:
            continue

        global_step = (micro_step + 1) // config.gradient_accumulation_steps
        scaler.unscale_(optimizer)
        grad_norm = grad_global_norm(model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        lr = scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if global_step == 1 or global_step % config.log_interval == 0:
            elapsed = max(1.0e-6, time.perf_counter() - window_start)
            metrics = {
                "step": global_step,
                "loss": round(running_loss / max(1, running_loss_count), 6),
                "lr": lr,
                "examples": running_examples,
                "target_tokens": running_target_tokens,
                "target_tokens_per_sec": round(running_target_tokens / elapsed, 2),
                "grad_norm": round(grad_norm, 6),
                "memory_mb": round(current_memory_mb(device), 2),
            }
            logger.log(metrics)
            print(
                f"step {global_step:04d} | loss {metrics['loss']:.4f} | "
                f"lr {lr:.2e} | target tok/s {metrics['target_tokens_per_sec']:.1f} | grad {metrics['grad_norm']:.3f}"
            )
            running_loss = 0.0
            running_loss_count = 0
            running_target_tokens = 0
            running_examples = 0
            window_start = time.perf_counter()

        if global_step % config.save_interval == 0 or global_step == config.max_steps:
            step_path, latest_path = save_sft_checkpoint(
                config.output_dir,
                global_step,
                model,
                optimizer,
                scheduler,
                config,
                model_config,
                scaler,
            )
            print(f"saved checkpoint: {step_path}")
            print(f"updated latest: {latest_path}")


def dry_run(config: SFTConfig, limit_examples: int | None) -> None:
    tokenizer = load_tokenizer(config.tokenizer_path, allow_byte_fallback=False)
    dataset = NextActionSFTDataset(config.train_data_path, tokenizer, config.max_seq_len, limit=limit_examples)
    checkpoint = torch.load(config.base_checkpoint, map_location="cpu", weights_only=False)
    model_config = MiniMindConfig.from_dict(checkpoint["model_config"])
    first = dataset[0]
    supervised = int((first["labels"] != IGNORE_INDEX).sum().item())
    print(f"run_name: {config.run_name}")
    print(f"base_checkpoint: {config.base_checkpoint}")
    print(f"train_data_path: {config.train_data_path}")
    print(f"examples: {len(dataset)}")
    print(f"tokenizer vocab: {tokenizer.vocab_size}")
    print(f"model vocab: {model_config.vocab_size}")
    print(f"batch/effective_batch: {config.batch_size}/{config.effective_batch_size}")
    print(f"max_seq_len: {config.max_seq_len}")
    print(f"first example length: {first['input_ids'].numel()}")
    print(f"first example supervised tokens: {supervised}")
    print(f"dtype/device: {config.dtype}/{config.device}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT a MiniMind checkpoint on tool-use next-action data.")
    parser.add_argument("--config", default="configs/sft_minimind_webnav_smoke.yaml")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-examples", type=int, default=None)
    args = parser.parse_args()

    config = SFTConfig.from_yaml(args.config, overrides=args.override)
    if args.dry_run:
        dry_run(config, limit_examples=args.limit_examples)
        return
    train(args, config)


if __name__ == "__main__":
    main()
