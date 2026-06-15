"""Inspect model shape, parameter count, and a tiny forward/backward pass."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.model import MiniMindConfig, MiniMindForCausalLM
from src.utils.param_count import count_parameters, format_parameter_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a MiniMind model config.")
    parser.add_argument("--config", default="configs/minimind_64m.yaml", help="Path to model YAML config.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    args = parser.parse_args()

    config = MiniMindConfig.from_yaml(args.config)
    model = MiniMindForCausalLM(config)
    model.train()

    input_ids = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    if output.loss is None:
        raise RuntimeError("Expected loss when labels are provided.")
    output.loss.backward()

    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)

    print(f"config: {args.config}")
    print(f"hidden_size/layers/heads: {config.hidden_size}/{config.num_hidden_layers}/{config.num_attention_heads}")
    print(f"kv_heads/head_dim: {config.num_key_value_heads}/{config.head_dim}")
    print(f"logits shape: {tuple(output.logits.shape)}")
    print(f"loss: {output.loss.item():.4f}")
    print(f"total params: {total_params} ({format_parameter_count(total_params)})")
    print(f"trainable params: {trainable_params} ({format_parameter_count(trainable_params)})")


if __name__ == "__main__":
    main()
