"""Text generation entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data import load_tokenizer
from src.model import MiniMindConfig, MiniMindForCausalLM


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> torch.Tensor:
    """Sample one token from final-step logits."""
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
        logits = torch.where(logits < values[:, [-1]], torch.full_like(logits, float("-inf")), logits)

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(dim=-1, index=sorted_indices, src=sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(
    model: MiniMindForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    eos_token_id: int | None,
) -> torch.Tensor:
    """Autoregressively generate tokens without KV cache."""
    generated = input_ids
    max_context = model.config.max_position_embeddings

    for _ in range(max_new_tokens):
        context_ids = generated[:, -max_context:]
        attention_mask = torch.ones_like(context_ids)
        output = model(input_ids=context_ids, attention_mask=attention_mask)
        next_logits = output.logits[:, -1, :]
        next_token = sample_next_token(next_logits, temperature=temperature, top_k=top_k, top_p=top_p)
        generated = torch.cat([generated, next_token], dim=-1)

        if eos_token_id is not None and int(next_token.item()) == eos_token_id:
            break

    return generated


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Generate text from a MiniMind checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file.")
    parser.add_argument("--prompt", default="MiniMind", help="Prompt text. Ignored when --prompt-file is set.")
    parser.add_argument("--prompt-file", default=None, help="Optional UTF-8 text file to use as the prompt.")
    parser.add_argument("--tokenizer-path", default=None, help="Tokenizer path. Defaults to checkpoint train config.")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or auto.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--output", default="outputs/generated_sample.txt")
    parser.add_argument("--allow-byte-fallback", action="store_true", default=True)
    parser.add_argument("--no-byte-fallback", action="store_false", dest="allow_byte_fallback")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = MiniMindConfig.from_dict(checkpoint["model_config"])
    train_config = checkpoint.get("train_config", {})
    tokenizer_path = args.tokenizer_path or train_config.get("tokenizer_path", "data/tokenizer")

    tokenizer = load_tokenizer(tokenizer_path, allow_byte_fallback=args.allow_byte_fallback)
    model = MiniMindForCausalLM(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prompt = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else args.prompt
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("Prompt produced no tokens.")
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    generated_ids = generate(
        model,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=tokenizer.eos_token_id,
    )
    token_ids = generated_ids[0].tolist()
    text = tokenizer.decode(token_ids)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")

    print(f"checkpoint: {args.checkpoint}")
    print(f"step: {checkpoint.get('step')}")
    print(f"tokenizer: {type(tokenizer).__name__}")
    print(f"prompt: {prompt}")
    print("generated:")
    print(text)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
