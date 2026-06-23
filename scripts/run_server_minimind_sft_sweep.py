"""Server-side MiniMind SFT and rollout sweep runner."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str], dry_run: bool) -> None:
    print("\n$", " ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True)


def parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected a comma-separated integer list.")
    return [int(item) for item in items]


def write_summary(report_paths: list[Path], output_path: Path) -> None:
    rows = []
    for path in report_paths:
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        summary = report.get("summary", {})
        rows.append(
            {
                "run": path.stem,
                "checkpoint": report.get("checkpoint"),
                "limit": report.get("limit"),
                "success_rate": summary.get("success_rate"),
                "success": summary.get("success"),
                "total": summary.get("total"),
                "submitted_rate": summary.get("submitted_rate"),
                "format_errors": summary.get("format_errors"),
                "invalid_tool_calls": summary.get("invalid_tool_calls"),
                "avg_steps": summary.get("avg_steps"),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nsummary: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MiniMind server SFT/rollout sweep.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--config", default="configs/sft_minimind_webnav_smoke.yaml")
    parser.add_argument("--epochs", type=parse_int_list, default=[1, 2, 3])
    parser.add_argument("--steps-per-epoch", type=int, default=320)
    parser.add_argument("--eval-limits", type=parse_int_list, default=[20, 50, 200])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--webnav-root", default="D:/job/Program/WebNav-RL")
    parser.add_argument("--tasks", default="D:/job/Program/WebNav-RL/tasks/eval_tasks.jsonl")
    parser.add_argument("--run-prefix", default="sft_minimind_webnav")
    parser.add_argument("--checkpoint-root", default="checkpoints")
    parser.add_argument("--report-root", default="reports/server_sweep")
    parser.add_argument("--trajectory-root", default="outputs/server_sweep")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-every-epoch", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    report_paths: list[Path] = []
    for epoch in args.epochs:
        max_steps = epoch * args.steps_per_epoch
        warmup_steps = max(1, max_steps // 20)
        run_name = f"{args.run_prefix}_epoch{epoch}"
        output_dir = Path(args.checkpoint_root) / run_name
        save_interval = args.steps_per_epoch if args.save_every_epoch else max_steps

        if not args.skip_train:
            train_command = [
                args.python,
                "scripts/train_sft_minimind.py",
                "--config",
                args.config,
                "--override",
                f"run_name={run_name}",
                "--override",
                f"output_dir={output_dir.as_posix()}",
                "--override",
                f"max_steps={max_steps}",
                "--override",
                f"warmup_steps={warmup_steps}",
                "--override",
                f"log_interval={args.log_interval}",
                "--override",
                f"save_interval={save_interval}",
                "--override",
                f"device={args.device}",
            ]
            run_command(train_command, dry_run=args.dry_run)

        if args.skip_eval:
            continue

        checkpoint = output_dir / "latest.pt"
        for limit in args.eval_limits:
            report_path = Path(args.report_root) / f"{run_name}_eval{limit}.json"
            output_path = Path(args.trajectory_root) / f"{run_name}_eval{limit}_trajectories.jsonl"
            eval_command = [
                args.python,
                "scripts/run_minimind_webnav_eval.py",
                "--checkpoint",
                checkpoint.as_posix(),
                "--webnav-root",
                args.webnav_root,
                "--tasks",
                args.tasks,
                "--limit",
                str(limit),
                "--device",
                args.device,
                "--max-new-tokens",
                "48",
                "--output",
                output_path.as_posix(),
                "--report",
                report_path.as_posix(),
            ]
            run_command(eval_command, dry_run=args.dry_run)
            report_paths.append(report_path)

    if not args.dry_run and report_paths:
        write_summary(report_paths, Path(args.report_root) / "summary.json")


if __name__ == "__main__":
    main()
