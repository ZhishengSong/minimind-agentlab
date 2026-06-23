"""Summarize rollout report JSON files into JSON and Markdown tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MiniMind rollout reports.")
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--output-json", default="reports/server_sweep/summary.json")
    parser.add_argument("--output-md", default="reports/server_sweep/summary.md")
    args = parser.parse_args()

    rows = []
    for pattern in args.reports:
        paths = sorted(Path().glob(pattern))
        if not paths and Path(pattern).exists():
            paths = [Path(pattern)]
        for path in paths:
            report = json.loads(path.read_text(encoding="utf-8"))
            summary = report.get("summary", {})
            rows.append(
                {
                    "report": path.as_posix(),
                    "checkpoint": report.get("checkpoint"),
                    "limit": report.get("limit"),
                    "success": summary.get("success"),
                    "total": summary.get("total"),
                    "success_rate": summary.get("success_rate"),
                    "submitted_rate": summary.get("submitted_rate"),
                    "format_errors": summary.get("format_errors"),
                    "invalid_tool_calls": summary.get("invalid_tool_calls"),
                    "avg_steps": summary.get("avg_steps"),
                }
            )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# MiniMind Rollout Summary",
        "",
        "| Report | Limit | Success | Success Rate | Submitted Rate | Format Errors | Invalid Calls | Avg Steps |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {report} | {limit} | {success}/{total} | {success_rate} | {submitted_rate} | {format_errors} | {invalid_tool_calls} | {avg_steps} |".format(
                **row
            )
        )
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved: {output_json}")
    print(f"saved: {output_md}")


if __name__ == "__main__":
    main()
