"""Validate tokenizer and JSONL assets before real pretraining."""

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
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Validate real pretraining assets.")
    parser.add_argument("--config", default="configs/pretrain_minimind_local.yaml")
    parser.add_argument("--num-examples", type=int, default=4)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--allow-byte-fallback", action="store_true", default=False)
    args = parser.parse_args()

    train_config = PretrainConfig.from_yaml(args.config)
    model_config = train_config.load_model_config()
    try:
        tokenizer = load_tokenizer(train_config.tokenizer_path, allow_byte_fallback=args.allow_byte_fallback)
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        print("Place tokenizer files under data/minimind/tokenizer or update tokenizer_path in the config.")
        raise SystemExit(1) from exc

    if tokenizer.vocab_size != model_config.vocab_size:
        print(
            "warning: tokenizer vocab size "
            f"({tokenizer.vocab_size}) != model vocab size ({model_config.vocab_size})"
        )

    try:
        dataset = PretrainDataset(
            train_config.train_data_path,
            tokenizer=tokenizer,
            max_seq_len=train_config.max_seq_len,
            text_field=args.text_field,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        print("Place pretrain_t2t_mini.jsonl under data/minimind or update train_data_path in the config.")
        raise SystemExit(1) from exc
    examples = list(islice(iter(dataset), args.num_examples))
    if not examples:
        raise ValueError(f"No examples found in {train_config.train_data_path}")

    collator = CausalLMCollator(pad_token_id=tokenizer.pad_token_id)
    batch = collator(examples)

    print(f"config: {args.config}")
    print(f"tokenizer_path: {train_config.tokenizer_path}")
    print(f"train_data_path: {train_config.train_data_path}")
    print(f"tokenizer_type: {type(tokenizer).__name__}")
    print(f"tokenizer_vocab_size: {tokenizer.vocab_size}")
    print(f"model_vocab_size: {model_config.vocab_size}")
    print(f"num_examples_checked: {len(examples)}")
    print(f"input_ids shape: {tuple(batch['input_ids'].shape)}")
    print(f"attention lengths: {batch['attention_mask'].sum(dim=1).tolist()}")
    print(f"ignored labels: {(batch['labels'] == -100).sum().item()}")
    print("first decoded sample:")
    print(tokenizer.decode(examples[0]["input_ids"][:200]))


if __name__ == "__main__":
    main()
