"""Model configuration definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.utils.config import load_yaml, save_yaml


@dataclass(slots=True)
class MiniMindConfig:
    """Configuration for a MiniMind-style causal language model."""

    model_type: str = "minimind"
    vocab_size: int = 6400
    hidden_size: int = 512
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    intermediate_size: int = 1408
    max_position_embeddings: int = 2048
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1.0e-6
    tie_word_embeddings: bool = True
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.model_type != "minimind":
            raise ValueError(f"Unsupported model_type: {self.model_type!r}")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive.")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive.")
        if self.num_key_value_heads <= 0:
            raise ValueError("num_key_value_heads must be positive.")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads.")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads.")
        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive.")
        if self.max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive.")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive.")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MiniMindConfig":
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MiniMindConfig":
        return cls.from_dict(load_yaml(path))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_yaml(self, path: str | Path) -> None:
        save_yaml(self.to_dict(), path)
