"""Create a tool-use tokenizer and resized MiniMind init checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tokenizers import AddedToken, Tokenizer


DEFAULT_BASE_TOKENS = [
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
    "<|im_start|>",
    "<|im_end|>",
]


def load_special_tokens(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    tokens = data.get("required_special_tokens")
    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
        raise ValueError(f"{path} must contain a string list at required_special_tokens.")
    return list(dict.fromkeys(tokens))


def copy_tokenizer(src_dir: Path, dst_dir: Path) -> None:
    if not src_dir.exists():
        raise FileNotFoundError(f"Tokenizer directory not found: {src_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    for path in src_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, dst_dir / path.name)


def update_tokenizer_config(config_path: Path, tokens: list[str], token_to_id: dict[str, int]) -> None:
    if not config_path.exists():
        return

    config = json.loads(config_path.read_text(encoding="utf-8"))
    decoder = config.setdefault("added_tokens_decoder", {})
    for token in tokens:
        token_id = str(token_to_id[token])
        decoder[token_id] = {
            "content": token,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        }

    existing = list(config.get("additional_special_tokens", []))
    for token in tokens:
        if token not in existing:
            existing.append(token)
    config["additional_special_tokens"] = existing
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resize_embedding(
    weight: torch.Tensor,
    new_vocab_size: int,
    init_ids: list[int],
    generator: torch.Generator,
    noise_std: float,
) -> torch.Tensor:
    old_vocab_size, hidden_size = weight.shape
    if new_vocab_size < old_vocab_size:
        raise ValueError("new_vocab_size cannot be smaller than old_vocab_size.")
    if new_vocab_size == old_vocab_size:
        return weight.clone()

    resized = weight.new_empty((new_vocab_size, hidden_size))
    resized[:old_vocab_size] = weight
    base = weight[init_ids].mean(dim=0) if init_ids else weight.mean(dim=0)
    for row in range(old_vocab_size, new_vocab_size):
        noise = torch.randn(hidden_size, generator=generator, dtype=weight.dtype) * noise_std
        resized[row] = base + noise
    return resized


def summarize_tokens(tokenizer: Tokenizer, tokens: list[str]) -> list[dict[str, object]]:
    summary = []
    for token in tokens:
        token_id = tokenizer.token_to_id(token)
        summary.append(
            {
                "token": token,
                "id": token_id,
                "encoded_ids": tokenizer.encode(token).ids,
                "is_single_token": token_id is not None and tokenizer.encode(token).ids == [token_id],
            }
        )
    return summary


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Prepare MiniMind checkpoint for tool-use SFT.")
    parser.add_argument("--base-checkpoint", default="minimind-50k-artifacts/checkpoints/pretrain_step_050000.pt")
    parser.add_argument("--base-tokenizer", default="data/minimind/tokenizer")
    parser.add_argument("--special-tokens", default="configs/tool_special_tokens.json")
    parser.add_argument("--output-tokenizer", default="outputs/tooluse_init/tokenizer")
    parser.add_argument("--output-checkpoint", default="outputs/tooluse_init/pretrain_step_050000_tooluse_init.pt")
    parser.add_argument("--report", default="reports/tooluse_init_report.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-std", type=float, default=0.001)
    args = parser.parse_args()

    base_checkpoint = Path(args.base_checkpoint)
    base_tokenizer = Path(args.base_tokenizer)
    output_tokenizer = Path(args.output_tokenizer)
    output_checkpoint = Path(args.output_checkpoint)
    report_path = Path(args.report)

    special_tokens = load_special_tokens(Path(args.special_tokens))
    copy_tokenizer(base_tokenizer, output_tokenizer)

    tokenizer_json = output_tokenizer / "tokenizer.json"
    if not tokenizer_json.exists():
        raise FileNotFoundError(f"tokenizer.json not found in {output_tokenizer}")

    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    old_vocab_size = tokenizer.get_vocab_size()
    existing_tokens = [token for token in special_tokens if tokenizer.token_to_id(token) is not None]
    missing_tokens = [token for token in special_tokens if tokenizer.token_to_id(token) is None]
    added = tokenizer.add_special_tokens(
        [
            AddedToken(token, single_word=False, lstrip=False, rstrip=False, normalized=False, special=True)
            for token in missing_tokens
        ]
    )
    new_vocab_size = tokenizer.get_vocab_size()
    tokenizer.save(str(tokenizer_json), pretty=True)

    token_to_id = {token: tokenizer.token_to_id(token) for token in special_tokens}
    update_tokenizer_config(output_tokenizer / "tokenizer_config.json", special_tokens, token_to_id)

    checkpoint = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    model_config: dict[str, Any] = dict(checkpoint["model_config"])
    train_config: dict[str, Any] = dict(checkpoint.get("train_config", {}))
    state_dict = dict(checkpoint["model_state_dict"])

    old_model_vocab_size = int(model_config["vocab_size"])
    if old_vocab_size != old_model_vocab_size:
        raise ValueError(f"Tokenizer vocab ({old_vocab_size}) != checkpoint vocab ({old_model_vocab_size}).")
    if int(state_dict["embed_tokens.weight"].shape[0]) != old_model_vocab_size:
        raise ValueError("Checkpoint embedding shape does not match model_config vocab_size.")

    init_ids = [tokenizer.token_to_id(token) for token in DEFAULT_BASE_TOKENS if tokenizer.token_to_id(token) is not None]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    resized = resize_embedding(
        state_dict["embed_tokens.weight"],
        new_vocab_size=new_vocab_size,
        init_ids=init_ids,
        generator=generator,
        noise_std=args.noise_std,
    )
    state_dict["embed_tokens.weight"] = resized
    state_dict["lm_head.weight"] = resized.clone()

    model_config["vocab_size"] = new_vocab_size
    train_config["tokenizer_path"] = str(output_tokenizer).replace("\\", "/")

    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": checkpoint.get("step"),
        "model_state_dict": state_dict,
        "model_config": model_config,
        "train_config": train_config,
        "base_checkpoint": str(base_checkpoint).replace("\\", "/"),
        "special_tokens": {
            "source": args.special_tokens,
            "existing_tokens": existing_tokens,
            "added_tokens": missing_tokens,
            "added_count": added,
            "old_vocab_size": old_vocab_size,
            "new_vocab_size": new_vocab_size,
            "init_base_tokens": DEFAULT_BASE_TOKENS,
            "init_noise_std": args.noise_std,
        },
    }
    torch.save(payload, output_checkpoint)

    report = {
        "base_checkpoint": str(base_checkpoint).replace("\\", "/"),
        "output_checkpoint": str(output_checkpoint).replace("\\", "/"),
        "base_tokenizer": str(base_tokenizer).replace("\\", "/"),
        "output_tokenizer": str(output_tokenizer).replace("\\", "/"),
        "old_vocab_size": old_vocab_size,
        "new_vocab_size": new_vocab_size,
        "existing_tokens": existing_tokens,
        "added_tokens": missing_tokens,
        "added_count": added,
        "token_summary": summarize_tokens(tokenizer, special_tokens),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
