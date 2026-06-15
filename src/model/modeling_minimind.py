"""MiniMind-style causal language model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from src.model.attention import MiniMindAttention
from src.model.config import MiniMindConfig
from src.model.mlp import MiniMindMLP
from src.model.norm import RMSNorm


@dataclass(slots=True)
class CausalLMOutput:
    """Output container for causal language modeling."""

    loss: torch.Tensor | None
    logits: torch.Tensor


class MiniMindBlock(nn.Module):
    """Pre-norm Transformer block."""

    def __init__(self, config: MiniMindConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = MiniMindAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MiniMindMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask, position_ids=position_ids)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class MiniMindForCausalLM(nn.Module):
    """MiniMind-style causal language model."""

    def __init__(self, config: MiniMindConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([MiniMindBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> CausalLMOutput:
        hidden_states = self.embed_tokens(input_ids)
        expanded_attention_mask = self._prepare_attention_mask(attention_mask, input_ids.shape, input_ids.device)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=expanded_attention_mask,
                position_ids=position_ids,
            )

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutput(loss=loss, logits=logits)

    def _prepare_attention_mask(
        self,
        attention_mask: torch.Tensor | None,
        input_shape: torch.Size,
        device: torch.device,
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None

        batch_size, seq_len = input_shape
        causal_mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device).tril()
        padding_mask = attention_mask.to(torch.bool).view(batch_size, 1, 1, seq_len)
        return causal_mask.view(1, 1, seq_len, seq_len) & padding_mask
