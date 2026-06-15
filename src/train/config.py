"""Training configuration definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.model.config import MiniMindConfig
from src.utils.config import apply_overrides, load_yaml, save_yaml


@dataclass(slots=True)
class PretrainConfig:
    """Configuration for base-model pretraining."""

    run_name: str = "pretrain_minimind_64m"
    model_config: str = "configs/minimind_64m.yaml"
    tokenizer_path: str = "data/tokenizer"
    train_data_path: str = "data/pretrain_t2t_mini.jsonl"
    output_dir: str = "checkpoints/pretrain"
    log_dir: str = "logs"

    seed: int = 42
    device: str = "auto"
    dtype: str = "bf16"

    max_seq_len: int = 2048
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_steps: int = 1000

    learning_rate: float = 3.0e-4
    min_learning_rate: float = 3.0e-5
    weight_decay: float = 0.1
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    log_interval: int = 10
    save_interval: int = 500
    sample_interval: int = 500
    num_workers: int = 2

    def __post_init__(self) -> None:
        if not self.run_name:
            raise ValueError("run_name cannot be empty.")
        if self.dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError("dtype must be one of: bf16, fp16, fp32.")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive.")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.min_learning_rate < 0:
            raise ValueError("min_learning_rate cannot be negative.")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate cannot exceed learning_rate.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative.")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive.")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive.")
        if self.save_interval <= 0:
            raise ValueError("save_interval must be positive.")
        if self.sample_interval <= 0:
            raise ValueError("sample_interval must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PretrainConfig":
        return cls(**data)

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        overrides: list[str] | None = None,
    ) -> "PretrainConfig":
        data = apply_overrides(load_yaml(path), overrides)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_yaml(self, path: str | Path) -> None:
        save_yaml(self.to_dict(), path)

    def load_model_config(self) -> MiniMindConfig:
        return MiniMindConfig.from_yaml(self.model_config)
