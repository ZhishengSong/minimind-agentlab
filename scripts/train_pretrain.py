"""Pretraining entry point."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.data import CausalLMCollator, PretrainDataset, load_tokenizer
from src.model import MiniMindForCausalLM
from src.train.checkpoint import load_checkpoint, save_checkpoint, save_config_copies
from src.train.config import PretrainConfig
from src.train.logger import JsonlLogger
from src.train.optim import WarmupCosineScheduler, build_optimizer, grad_global_norm
from src.utils.param_count import count_parameters, format_parameter_count


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


def train(args: argparse.Namespace, train_config: PretrainConfig) -> None:
    torch.manual_seed(train_config.seed)
    device = resolve_device(train_config.device)
    model_config = train_config.load_model_config()
    tokenizer = load_tokenizer(train_config.tokenizer_path, allow_byte_fallback=args.allow_byte_fallback)
    if tokenizer.vocab_size > model_config.vocab_size:
        raise ValueError(
            f"Tokenizer vocab size ({tokenizer.vocab_size}) exceeds model vocab size ({model_config.vocab_size})."
        )

    dataset = PretrainDataset(
        train_config.train_data_path,
        tokenizer=tokenizer,
        max_seq_len=train_config.max_seq_len,
    )
    collator = CausalLMCollator(pad_token_id=tokenizer.pad_token_id)
    dataloader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        collate_fn=collator,
        num_workers=train_config.num_workers,
    )

    model = MiniMindForCausalLM(model_config).to(device)
    model.train()

    optimizer = build_optimizer(model, train_config.learning_rate, train_config.weight_decay)
    scheduler = WarmupCosineScheduler(
        optimizer,
        max_steps=train_config.max_steps,
        warmup_steps=train_config.warmup_steps,
        learning_rate=train_config.learning_rate,
        min_learning_rate=train_config.min_learning_rate,
    )
    logger = JsonlLogger(train_config.log_dir, train_config.run_name)

    autocast_dtype = resolve_autocast_dtype(train_config.dtype)
    use_autocast = autocast_dtype is not None and device.type in {"cuda", "cpu"}
    scaler = torch.amp.GradScaler("cuda", enabled=train_config.dtype == "fp16" and device.type == "cuda")
    save_config_copies(train_config.output_dir, train_config, model_config)

    start_step = 0
    if args.resume:
        checkpoint = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        start_step = int(checkpoint["step"])
        print(f"resumed from {args.resume} at step {start_step}")

    print(f"run_name: {train_config.run_name}")
    print(f"device/dtype: {device}/{train_config.dtype}")
    print(f"model params: {format_parameter_count(count_parameters(model))}")
    print(f"tokenizer: {type(tokenizer).__name__}, vocab_size={tokenizer.vocab_size}")
    print(f"metrics: {logger.path}")
    print(f"output_dir: {train_config.output_dir}")

    data_iter = infinite_batches(dataloader)
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    running_tokens = 0
    running_updates = 0
    window_start = time.perf_counter()

    start_micro_step = start_step * train_config.gradient_accumulation_steps
    total_micro_steps = train_config.max_steps * train_config.gradient_accumulation_steps

    for micro_step in range(start_micro_step, total_micro_steps):
        batch = next(data_iter)
        batch = {key: value.to(device) for key, value in batch.items()}

        context = torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast)
        with context:
            output = model(**batch)
            if output.loss is None:
                raise RuntimeError("Model did not return loss during training.")
            loss = output.loss / train_config.gradient_accumulation_steps

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at micro_step {micro_step}: {loss.item()}")

        scaler.scale(loss).backward()
        running_loss += float(output.loss.detach().item())
        running_tokens += int(batch["attention_mask"].sum().item())

        is_update_step = (micro_step + 1) % train_config.gradient_accumulation_steps == 0
        if not is_update_step:
            continue

        global_step = (micro_step + 1) // train_config.gradient_accumulation_steps
        scaler.unscale_(optimizer)
        grad_norm = grad_global_norm(model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        lr = scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        running_updates += 1

        should_log = global_step == 1 or global_step % train_config.log_interval == 0
        if should_log:
            elapsed = max(1.0e-6, time.perf_counter() - window_start)
            avg_loss = running_loss / max(1, running_updates)
            tokens_per_sec = running_tokens / elapsed
            metrics = {
                "step": global_step,
                "loss": round(avg_loss, 6),
                "lr": lr,
                "tokens": running_tokens,
                "tokens_per_sec": round(tokens_per_sec, 2),
                "grad_norm": round(grad_norm, 6),
                "memory_mb": round(current_memory_mb(device), 2),
            }
            logger.log(metrics)
            print(
                f"step {global_step:04d} | loss {metrics['loss']:.4f} | "
                f"lr {lr:.2e} | tok/s {metrics['tokens_per_sec']:.1f} | grad {metrics['grad_norm']:.3f}"
            )
            running_loss = 0.0
            running_tokens = 0
            running_updates = 0
            window_start = time.perf_counter()

        should_save = global_step % train_config.save_interval == 0 or global_step == train_config.max_steps
        if should_save:
            step_path, latest_path = save_checkpoint(
                train_config.output_dir,
                step=global_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_config=train_config,
                model_config=model_config,
                scaler=scaler,
            )
            print(f"saved checkpoint: {step_path}")
            print(f"updated latest: {latest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain a MiniMind-style causal LM.")
    parser.add_argument("--config", default="configs/pretrain.yaml", help="Path to pretraining YAML config.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config value, e.g. --override batch_size=2 --override dtype=fp32.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate configs, then print a summary without training.",
    )
    parser.add_argument(
        "--allow-byte-fallback",
        action="store_true",
        default=True,
        help="Use a simple byte tokenizer when tokenizer_path is missing.",
    )
    parser.add_argument("--no-byte-fallback", action="store_false", dest="allow_byte_fallback")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from.")
    args = parser.parse_args()

    train_config = PretrainConfig.from_yaml(args.config, overrides=args.override)
    model_config = train_config.load_model_config()

    if args.dry_run:
        print(f"run_name: {train_config.run_name}")
        print(f"pretrain_config: {args.config}")
        print(f"model_config: {train_config.model_config}")
        print(f"model hidden_size/layers/heads: {model_config.hidden_size}/{model_config.num_hidden_layers}/{model_config.num_attention_heads}")
        print(f"kv_heads: {model_config.num_key_value_heads}, head_dim: {model_config.head_dim}")
        print(f"batch_size: {train_config.batch_size}")
        print(f"gradient_accumulation_steps: {train_config.gradient_accumulation_steps}")
        print(f"effective_batch_size: {train_config.effective_batch_size}")
        print(f"dtype/device: {train_config.dtype}/{train_config.device}")
        return

    train(args, train_config)


if __name__ == "__main__":
    main()
