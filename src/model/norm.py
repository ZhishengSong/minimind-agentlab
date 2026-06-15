"""RMSNorm implementation."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root mean square layer normalization.

    RMSNorm normalizes by the root mean square of the hidden dimension and
    applies a learned scale. It does not subtract the mean.
    """

    def __init__(self, hidden_size: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).to(input_dtype)
