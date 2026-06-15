"""Optimizer and learning-rate scheduler helpers."""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn


def get_parameter_groups(
    model: nn.Module,
    weight_decay: float,
) -> list[dict[str, object]]:
    """Split parameters into decay and no-decay groups."""
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or name.endswith(".bias") or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def build_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Build an AdamW optimizer."""
    return torch.optim.AdamW(
        get_parameter_groups(model, weight_decay),
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1.0e-8,
    )


class WarmupCosineScheduler:
    """Step-wise warmup plus cosine decay scheduler."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_steps: int,
        warmup_steps: int,
        learning_rate: float,
        min_learning_rate: float,
    ) -> None:
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.step_num = 0
        self._set_lr(self.get_lr(0))

    def get_lr(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.learning_rate * float(step + 1) / float(self.warmup_steps)

        if step >= self.max_steps:
            return self.min_learning_rate

        decay_steps = max(1, self.max_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, (step - self.warmup_steps) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_learning_rate + cosine * (self.learning_rate - self.min_learning_rate)

    def step(self) -> float:
        lr = self.get_lr(self.step_num)
        self._set_lr(lr)
        self.step_num += 1
        return lr

    def state_dict(self) -> dict[str, int | float]:
        return {"step_num": self.step_num}

    def load_state_dict(self, state_dict: dict[str, int | float]) -> None:
        self.step_num = int(state_dict["step_num"])
        self._set_lr(self.get_lr(self.step_num))

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr


def grad_global_norm(parameters: Iterable[nn.Parameter]) -> float:
    """Compute global gradient norm."""
    norms = [p.grad.detach().float().norm(2) for p in parameters if p.grad is not None]
    if not norms:
        return 0.0
    return float(torch.norm(torch.stack(norms), 2).item())
