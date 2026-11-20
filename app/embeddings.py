"""Milestone 2b: turn token ids into vectors.

A GPT sees each position as   token_embedding[token_id] + position_embedding[position].
Both tables are learned. The token table is also reused (weight tying) as the output layer of the
model, which saves parameters and helps small models.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from app.tokenizer import BPETokenizer


class GPTEmbedding(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_embd: int, dropout: float = 0.0):
        super().__init__()
        self.block_size = block_size
        self.token = nn.Embedding(vocab_size, n_embd)
        self.position = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        nn.init.normal_(self.token.weight, std=0.02)
        nn.init.normal_(self.position.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B, T) token ids  ->  (B, T, n_embd) vectors."""
        _, t = idx.shape
        if t > self.block_size:
            raise ValueError(f"sequence length {t} exceeds block_size {self.block_size}")
        pos = torch.arange(t, device=idx.device)
        return self.drop(self.token(idx) + self.position(pos))


def encode_corpus(tokenizer: BPETokenizer, text: str, cache: str | Path | None = None) -> torch.Tensor:
    """Encode the whole corpus into one long 1-D tensor of token ids (optionally cached on disk)."""
    if cache and Path(cache).exists():
        return torch.load(cache)
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        torch.save(ids, cache)
    return ids


def nearest_tokens(embedding: nn.Embedding, tokenizer: BPETokenizer, word: str, k: int = 8) -> list[str]:
    """Tokens whose (learned) embedding is closest by cosine similarity - handy to inspect a model."""
    ids = tokenizer.encode(word)
    with torch.no_grad():
        w = nn.functional.normalize(embedding.weight, dim=-1)
        sims = w @ w[ids[-1]]
        best = sims.topk(k + 1).indices.tolist()
    return [tokenizer.vocab[i].decode("utf-8", "replace") for i in best if i != ids[-1]][:k]
