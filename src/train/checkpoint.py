"""Checkpoint save and resume helpers."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.model.config import MiniMindConfig
from src.train.config import PretrainConfig
from src.train.optim import WarmupCosineScheduler
from src.utils.config import save_yaml


def get_random_state() -> dict[str, Any]:
    """Collect random states needed for reproducible resume."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_random_state(state: dict[str, Any]) -> None:
    """Restore saved random states."""
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([cuda_state.cpu() for cuda_state in state["cuda"]])


def checkpoint_path(output_dir: str | Path, step: int) -> Path:
    return Path(output_dir) / f"pretrain_step_{step:06d}.pt"


def latest_checkpoint_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "latest.pt"


def save_checkpoint(
    output_dir: str | Path,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    train_config: PretrainConfig,
    model_config: MiniMindConfig,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[Path, Path]:
    """Save a step checkpoint and update latest.pt."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    payload = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        "train_config": train_config.to_dict(),
        "model_config": model_config.to_dict(),
        "random_state": get_random_state(),
    }

    step_path = checkpoint_path(output_path, step)
    latest_path = latest_checkpoint_path(output_path)
    torch.save(payload, step_path)
    torch.save(payload, latest_path)
    return step_path, latest_path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: WarmupCosineScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
    restore_random_state: bool = True,
) -> dict[str, Any]:
    """Load checkpoint state into provided objects."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    if restore_random_state and checkpoint.get("random_state") is not None:
        set_random_state(checkpoint["random_state"])

    return checkpoint


def save_config_copies(
    output_dir: str | Path,
    train_config: PretrainConfig,
    model_config: MiniMindConfig,
) -> None:
    """Write resolved configs beside checkpoints."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_yaml(train_config.to_dict(), output_path / "pretrain_config.yaml")
    save_yaml(model_config.to_dict(), output_path / "model_config.yaml")
