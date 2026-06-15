"""Inspect one pretraining batch."""

from __future__ import annotations

import argparse
import sys
from itertools import islice
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import CausalLMCollator, PretrainDataset, load_tokenizer
from src.train.config import PretrainConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one pretraining batch.")
    parser.add_argument("--config", default="configs/pretrain_tiny.yaml", help="Path to pretraining YAML config.")
    parser.add_argument("--text-field", default="text", help="JSONL field containing raw text.")
    parser.add_argument(
        "--allow-byte-fallback",
        action="store_true",
        default=True,
        help="Use a simple byte tokenizer when tokenizer_path is missing.",
    )
    parser.add_argument("--no-byte-fallback", action="store_false", dest="allow_byte_fallback")
    args = parser.parse_args()

    train_config = PretrainConfig.from_yaml(args.config)
    tokenizer = load_tokenizer(train_config.tokenizer_path, allow_byte_fallback=args.allow_byte_fallback)
    dataset = PretrainDataset(
        train_config.train_data_path,
        tokenizer=tokenizer,
        max_seq_len=train_config.max_seq_len,
        text_field=args.text_field,
    )
    collator = CausalLMCollator(pad_token_id=tokenizer.pad_token_id)
    examples = list(islice(iter(dataset), train_config.batch_size))
    batch = collator(examples)

    print(f"config: {args.config}")
    print(f"data_path: {train_config.train_data_path}")
    print(f"tokenizer: {type(tokenizer).__name__}")
    print(f"vocab_size: {tokenizer.vocab_size}")
    print(f"pad_token_id: {tokenizer.pad_token_id}")
    print(f"eos_token_id: {tokenizer.eos_token_id}")
    print(f"num_examples: {len(examples)}")
    print(f"input_ids shape: {tuple(batch['input_ids'].shape)}")
    print(f"attention_mask shape: {tuple(batch['attention_mask'].shape)}")
    print(f"labels shape: {tuple(batch['labels'].shape)}")
    print(f"attention lengths: {batch['attention_mask'].sum(dim=1).tolist()}")
    print(f"num ignored labels: {(batch['labels'] == -100).sum().item()}")

    first_ids = batch["input_ids"][0].tolist()
    first_visible_ids = [
        token_id
        for token_id, visible in zip(first_ids, batch["attention_mask"][0].tolist(), strict=True)
        if visible
    ]
    print("first decoded sample:")
    print(tokenizer.decode(first_visible_ids))


if __name__ == "__main__":
    main()
