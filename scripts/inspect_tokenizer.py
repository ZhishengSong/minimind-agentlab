"""Inspect tokenizer assets and optionally compare with model config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_tokenizer
from src.model import MiniMindConfig


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Inspect tokenizer assets.")
    parser.add_argument("--tokenizer-path", default="data/minimind/tokenizer")
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--sample", default="MiniMind AgentLab tokenizer check.")
    parser.add_argument("--strict-vocab-match", action="store_true")
    parser.add_argument("--allow-byte-fallback", action="store_true", default=False)
    args = parser.parse_args()

    try:
        tokenizer = load_tokenizer(args.tokenizer_path, allow_byte_fallback=args.allow_byte_fallback)
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        print("Place tokenizer files under data/minimind/tokenizer or pass --tokenizer-path.")
        raise SystemExit(1) from exc
    token_ids = tokenizer.encode(args.sample, add_special_tokens=False)
    decoded = tokenizer.decode(token_ids)

    print(f"tokenizer_path: {args.tokenizer_path}")
    print(f"tokenizer_type: {type(tokenizer).__name__}")
    print(f"vocab_size: {tokenizer.vocab_size}")
    print(f"pad_token_id: {tokenizer.pad_token_id}")
    print(f"eos_token_id: {tokenizer.eos_token_id}")
    print(f"sample: {args.sample}")
    print(f"encoded_length: {len(token_ids)}")
    print(f"first_token_ids: {token_ids[:20]}")
    print(f"decoded: {decoded}")

    if args.model_config:
        model_config = MiniMindConfig.from_yaml(args.model_config)
        print(f"model_config: {args.model_config}")
        print(f"model_vocab_size: {model_config.vocab_size}")

        if model_config.vocab_size != tokenizer.vocab_size:
            message = (
                f"Tokenizer vocab size ({tokenizer.vocab_size}) does not match "
                f"model vocab size ({model_config.vocab_size})."
            )
            if args.strict_vocab_match:
                raise ValueError(message)
            print(f"warning: {message}")


if __name__ == "__main__":
    main()
