"""Milestone 5: generate story continuations by sampling from the trained GPT.

At each step the model gives a score (logit) for every token in the vocabulary. We
  1. divide by the temperature   (low = safe and repetitive, high = surprising and chaotic),
  2. lower the scores of tokens that were just used (repetition penalty),
  3. keep only the top-k tokens and/or the smallest set whose probability adds up to top-p,
  4. sample one token, append it, and repeat with the context cropped to block_size.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from app.model import GPT
from app.tokenizer import BPETokenizer

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


@dataclass
class SamplingConfig:
    max_new_tokens: int = 120
    temperature: float = 0.8
    top_k: int | None = 50
    top_p: float | None = 0.95
    repetition_penalty: float = 1.15
    repetition_window: int = 64
    stop_at_sentence: bool = True   # finish on . ! ? (or a closing quote after one) when possible
    seed: int | None = None


def filter_logits(logits: torch.Tensor, top_k: int | None = None, top_p: float | None = None) -> torch.Tensor:
    """Set the logits of tokens outside the top-k / nucleus (top-p) set to -inf. logits: (V,)."""
    logits = logits.clone()
    if top_k is not None and 0 < top_k < logits.size(-1):
        kth = torch.topk(logits, top_k).values[-1]
        logits[logits < kth] = float("-inf")
    if top_p is not None and 0 < top_p < 1:
        sorted_logits, order = torch.sort(logits, descending=True)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum > top_p
        remove[1:] = remove[:-1].clone()   # always keep the token that crosses the threshold
        remove[0] = False
        logits[order[remove]] = float("-inf")
    return logits


def apply_repetition_penalty(logits: torch.Tensor, recent: list[int], penalty: float) -> torch.Tensor:
    if penalty == 1.0 or not recent:
        return logits
    logits = logits.clone()
    idx = torch.tensor(sorted(set(recent)), device=logits.device)
    vals = logits[idx]
    logits[idx] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return logits


@torch.no_grad()
def sample(model: GPT, ids: list[int], cfg: SamplingConfig, ends_sentence=None) -> list[int]:
    """Return only the NEW token ids generated after `ids`."""
    if cfg.temperature <= 0:
        raise ValueError("temperature must be > 0")
    model.eval()
    device = next(model.parameters()).device
    g = torch.Generator(device="cpu").manual_seed(cfg.seed) if cfg.seed is not None else None
    context = list(ids) or [ord("\n")]
    new: list[int] = []
    budget = cfg.max_new_tokens + (40 if cfg.stop_at_sentence else 0)  # a little room to finish a sentence
    for i in range(budget):
        window = context[-model.cfg.block_size:]
        logits, _ = model(torch.tensor([window], device=device))
        logits = logits[0, -1].float().cpu() / cfg.temperature
        logits = apply_repetition_penalty(logits, context[-cfg.repetition_window:], cfg.repetition_penalty)
        logits = filter_logits(logits, cfg.top_k, cfg.top_p)
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=g).item()
        context.append(nxt)
        new.append(nxt)
        if i + 1 >= cfg.max_new_tokens and (not cfg.stop_at_sentence or (ends_sentence and ends_sentence(new))):
            break
    return new


class StoryGenerator:
    """Loads the tokenizer + trained model once and turns prompts into continuations."""

    def __init__(self, checkpoint_dir: str | Path = DEFAULT_DIR, device: str = "cpu"):
        checkpoint_dir = Path(checkpoint_dir)
        model_path = checkpoint_dir / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} not found: train first with `python -m app.cli train`")
        self.tokenizer = BPETokenizer.load(checkpoint_dir / "tokenizer.json")
        self.model, self.meta = GPT.load(model_path, device)

    @classmethod
    def from_parts(cls, model: GPT, tokenizer: BPETokenizer) -> "StoryGenerator":
        self = cls.__new__(cls)
        self.model, self.tokenizer, self.meta = model, tokenizer, {}
        return self

    def _ends_sentence(self, ids: list[int]) -> bool:
        tail = self.tokenizer.decode(ids[-3:]).rstrip()
        return tail.endswith((".", "!", "?", '."', '!"', '?"', ".'", "!'", "?'"))

    def continue_story(self, prompt: str, cfg: SamplingConfig | None = None) -> str:
        """Return only the generated continuation (not the prompt)."""
        cfg = cfg or SamplingConfig()
        ids = self.tokenizer.encode(prompt.rstrip(" "))  # a trailing space would split the next word
        new = sample(self.model, ids, cfg, self._ends_sentence)
        return self.tokenizer.decode(new)
