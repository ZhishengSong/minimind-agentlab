"""Training metric logging helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class JsonlLogger:
    """Append training metrics to a JSONL file."""

    def __init__(self, log_dir: str | Path, run_name: str) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / f"{run_name}_metrics.jsonl"
        self.start_time = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.start_time

    def log(self, metrics: dict[str, Any]) -> None:
        record = {"elapsed_sec": round(self.elapsed, 4), **metrics}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
