"""Iterable pretraining dataset."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from torch.utils.data import IterableDataset

from src.data.tokenizer import TokenizerLike


class PretrainDataset(IterableDataset[dict[str, list[int]]]):
    """Streaming JSONL dataset for causal LM pretraining."""

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: TokenizerLike,
        max_seq_len: int,
        text_field: str = "text",
        add_eos: bool = True,
    ) -> None:
        super().__init__()
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.text_field = text_field
        self.add_eos = add_eos

        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")
        if not self.data_path.exists():
            raise FileNotFoundError(f"Pretraining data not found: {self.data_path}")

    def __iter__(self) -> Iterator[dict[str, list[int]]]:
        with self.data_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {self.data_path}:{line_no}") from exc

                if self.text_field not in record:
                    raise KeyError(f"Missing field {self.text_field!r} in {self.data_path}:{line_no}")

                text = record[self.text_field]
                if not isinstance(text, str):
                    raise TypeError(f"Field {self.text_field!r} must be a string in {self.data_path}:{line_no}")

                token_ids = self.tokenizer.encode(text, add_special_tokens=False)
                if self.add_eos and self.tokenizer.eos_token_id is not None:
                    token_ids.append(self.tokenizer.eos_token_id)
                token_ids = token_ids[: self.max_seq_len]

                if token_ids:
                    yield {"input_ids": token_ids}
