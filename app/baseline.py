"""A baseline to measure the GPT against: a smoothed bigram model over the same BPE tokens.

It predicts the next token only from the current one (no long context, no attention). If the GPT
cannot beat this on held-out text, the transformer is not earning its keep.
"""
from __future__ import annotations

import math

import torch


class BigramBaseline:
    def __init__(self, vocab_size: int, alpha: float = 0.1):
        self.vocab_size, self.alpha = vocab_size, alpha
        self.counts = torch.zeros(vocab_size, vocab_size)

    def fit(self, ids: torch.Tensor) -> "BigramBaseline":
        pair_index = ids[:-1] * self.vocab_size + ids[1:]
        flat = torch.bincount(pair_index, minlength=self.vocab_size ** 2).float()
        self.counts = flat.view(self.vocab_size, self.vocab_size)
        return self

    def log_probs(self) -> torch.Tensor:
        smoothed = self.counts + self.alpha
        return torch.log(smoothed / smoothed.sum(dim=1, keepdim=True))

    def loss(self, ids: torch.Tensor) -> float:
        """Average cross-entropy (nats per token) on a token sequence."""
        lp = self.log_probs()
        return -lp[ids[:-1], ids[1:]].mean().item()

    def perplexity(self, ids: torch.Tensor) -> float:
        return math.exp(self.loss(ids))

    def generate(self, start: list[int], n: int, temperature: float = 1.0, seed: int | None = None) -> list[int]:
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        probs = torch.softmax(self.log_probs() / temperature, dim=1)
        out = list(start)
        for _ in range(n):
            out.append(torch.multinomial(probs[out[-1]], 1, generator=g).item())
        return out
