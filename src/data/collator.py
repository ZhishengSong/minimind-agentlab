"""Batch collation for causal language modeling."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class CausalLMCollator:
    """Pad variable-length token sequences into a causal LM batch."""

    pad_token_id: int
    label_pad_token_id: int = -100

    def __call__(self, examples: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        if not examples:
            raise ValueError("Cannot collate an empty batch.")

        max_len = max(len(example["input_ids"]) for example in examples)
        batch_size = len(examples)

        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((batch_size, max_len), self.label_pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

        for row, example in enumerate(examples):
            ids = torch.tensor(example["input_ids"], dtype=torch.long)
            seq_len = ids.numel()
            input_ids[row, :seq_len] = ids
            labels[row, :seq_len] = ids
            attention_mask[row, :seq_len] = 1

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
