"""Run the tiny local end-to-end validation flow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, command: list[str]) -> None:
    print(f"\n=== {name} ===", flush=True)
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tiny end-to-end validation.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use.")
    parser.add_argument("--train-steps", type=int, default=3)
    parser.add_argument("--resume-steps", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    py = args.python

    run_step(
        "Inspect config",
        [py, "scripts/train_pretrain.py", "--config", "configs/pretrain_tiny.yaml", "--dry-run"],
    )
    run_step(
        "Inspect model",
        [py, "scripts/inspect_model.py", "--config", "configs/minimind_tiny.yaml", "--batch-size", "1", "--seq-len", "16"],
    )
    run_step(
        "Inspect batch",
        [py, "scripts/inspect_batch.py", "--config", "configs/pretrain_tiny.yaml"],
    )
    run_step(
        "Train tiny",
        [
            py,
            "scripts/train_pretrain.py",
            "--config",
            "configs/pretrain_tiny.yaml",
            "--override",
            f"max_steps={args.train_steps}",
            "--override",
            "save_interval=2",
        ],
    )
    run_step(
        "Resume tiny",
        [
            py,
            "scripts/train_pretrain.py",
            "--config",
            "configs/pretrain_tiny.yaml",
            "--override",
            f"max_steps={args.resume_steps}",
            "--override",
            "save_interval=2",
            "--resume",
            "checkpoints/pretrain_tiny/latest.pt",
        ],
    )
    run_step(
        "Generate tiny",
        [
            py,
            "scripts/generate.py",
            "--checkpoint",
            "checkpoints/pretrain_tiny/latest.pt",
            "--prompt",
            "MiniMind",
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--temperature",
            "0",
        ],
    )

    print("\nTiny end-to-end validation completed.", flush=True)


if __name__ == "__main__":
    main()
