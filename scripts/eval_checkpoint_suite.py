"""Evaluate a suite of pretraining checkpoints on the same validation slice."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from argparse import Namespace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from scripts.eval_pretrain_loss import evaluate


DEFAULT_CHECKPOINTS = [
    "minimind-50k-artifacts/checkpoints/pretrain_step_005000.pt",
    "minimind-50k-artifacts/checkpoints/pretrain_step_010000.pt",
    "minimind-50k-artifacts/checkpoints/pretrain_step_020000.pt",
    "minimind-50k-artifacts/checkpoints/pretrain_step_050000.pt",
]


def write_markdown(results: list[dict[str, object]], output_path: Path) -> None:
    rows = [
        "| Checkpoint | Step | Loss | Perplexity | Predicted Tokens | Tokens/sec | Device |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        rows.append(
            "| {checkpoint} | {step} | {loss} | {ppl} | {tokens} | {toksec} | {device} |".format(
                checkpoint=Path(str(result["checkpoint"])).name,
                step=result.get("checkpoint_step", ""),
                loss=result.get("loss", ""),
                ppl=result.get("perplexity", ""),
                tokens=result.get("predicted_tokens", ""),
                toksec=result.get("tokens_per_sec", ""),
                device=result.get("device", ""),
            )
        )

    text = "\n".join(
        [
            "# Pretrain V0 Checkpoint Evaluation",
            "",
            "All checkpoints were evaluated on the same fixed JSONL slice.",
            "",
            *rows,
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Evaluate multiple MiniMind pretraining checkpoints.")
    parser.add_argument("--checkpoint", action="append", dest="checkpoints", help="Checkpoint path. Repeatable.")
    parser.add_argument("--config", default="configs/pretrain_minimind_local.yaml")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default=None, choices=["bf16", "fp16", "fp32", None])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--num-examples", type=int, default=1000)
    parser.add_argument("--skip-examples", type=int, default=0)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--save-subset", default=None)
    parser.add_argument("--output", default="reports/pretrain_v0_checkpoint_eval.json")
    parser.add_argument("--markdown-output", default="reports/pretrain_v0_checkpoint_eval.md")
    parser.add_argument("--add-eos", action="store_true", default=True)
    parser.add_argument("--no-add-eos", action="store_false", dest="add_eos")
    parser.add_argument("--allow-byte-fallback", action="store_true", default=False)
    args = parser.parse_args()

    checkpoint_paths = args.checkpoints or DEFAULT_CHECKPOINTS
    missing = [path for path in checkpoint_paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoint(s): " + ", ".join(missing))

    results: list[dict[str, object]] = []
    for index, checkpoint in enumerate(checkpoint_paths):
        save_subset = args.save_subset if index == 0 else None
        eval_args = Namespace(
            checkpoint=checkpoint,
            config=args.config,
            data_path=args.data_path,
            tokenizer_path=args.tokenizer_path,
            device=args.device,
            dtype=args.dtype,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            num_examples=args.num_examples,
            skip_examples=args.skip_examples,
            text_field=args.text_field,
            save_subset=save_subset,
            output=None,
            add_eos=args.add_eos,
            allow_byte_fallback=args.allow_byte_fallback,
        )
        print(f"evaluating: {checkpoint}")
        result = evaluate(eval_args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        results.append(result)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(results, Path(args.markdown_output))
    print(f"saved metrics: {output_path}")
    print(f"saved markdown: {args.markdown_output}")


if __name__ == "__main__":
    main()
