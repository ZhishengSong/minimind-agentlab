"""Tokenizer wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class TokenizerLike(Protocol):
    """Minimal tokenizer protocol used by the data pipeline."""

    pad_token_id: int
    eos_token_id: int | None
    vocab_size: int

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ...

    def decode(self, token_ids: list[int]) -> str:
        ...


class HFTokenizerWrapper:
    """Wrapper around Hugging Face tokenizers."""

    def __init__(self, tokenizer: object) -> None:
        self.tokenizer = tokenizer
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)

        if pad_token_id is None:
            if eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token
                pad_token_id = eos_token_id
            else:
                tokenizer.add_special_tokens({"pad_token": "<pad>"})
                pad_token_id = tokenizer.pad_token_id

        self.pad_token_id = int(pad_token_id)
        self.eos_token_id = int(eos_token_id) if eos_token_id is not None else None
        self.vocab_size = int(len(tokenizer))

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=add_special_tokens))

    def decode(self, token_ids: list[int]) -> str:
        return str(self.tokenizer.decode(token_ids, skip_special_tokens=False))


class TokenizersWrapper:
    """Wrapper around the low-level `tokenizers` package."""

    def __init__(self, tokenizer: object) -> None:
        self.tokenizer = tokenizer
        vocab = tokenizer.get_vocab()
        self.vocab_size = int(tokenizer.get_vocab_size())
        self.pad_token_id = int(vocab.get("<pad>", vocab.get("[PAD]", vocab.get("<|endoftext|>", 0))))
        eos = vocab.get("<eos>", vocab.get("</s>", vocab.get("[EOS]", vocab.get("<|im_end|>"))))
        self.eos_token_id = int(eos) if eos is not None else None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return list(encoding.ids)

    def decode(self, token_ids: list[int]) -> str:
        return str(self.tokenizer.decode(token_ids, skip_special_tokens=False))


@dataclass(slots=True)
class ByteTokenizer:
    """Small local fallback tokenizer for pipeline smoke tests.

    This is not the final MiniMind-compatible tokenizer. It exists so the data
    pipeline can be tested before the real tokenizer files are available.
    """

    pad_token_id: int = 0
    eos_token_id: int = 1
    vocab_size: int = 258

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        token_ids = [byte + 2 for byte in text.encode("utf-8")]
        if add_special_tokens:
            token_ids.append(self.eos_token_id)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        byte_values = [token_id - 2 for token_id in token_ids if 2 <= token_id <= 257]
        return bytes(byte_values).decode("utf-8", errors="replace")


def load_tokenizer(path: str | Path, allow_byte_fallback: bool = False) -> TokenizerLike:
    """Load a tokenizer from disk.

    The preferred path is a Hugging Face tokenizer directory. If the low-level
    `tokenizers` JSON file is provided, that is also supported.
    """
    tokenizer_path = Path(path)

    if tokenizer_path.exists():
        try:
            from transformers import AutoTokenizer

            return HFTokenizerWrapper(AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True))
        except Exception:
            pass

        tokenizer_json = tokenizer_path if tokenizer_path.is_file() else tokenizer_path / "tokenizer.json"
        if tokenizer_json.exists():
            from tokenizers import Tokenizer

            return TokenizersWrapper(Tokenizer.from_file(str(tokenizer_json)))

    if allow_byte_fallback:
        return ByteTokenizer()

    raise FileNotFoundError(
        f"Tokenizer not found at {tokenizer_path}. Provide a tokenizer path or use allow_byte_fallback=True."
    )
