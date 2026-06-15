"""SwiGLU feed-forward network implementation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.model.config import MiniMindConfig


class MiniMindMLP(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, config: MiniMindConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
