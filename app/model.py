"""Milestone 3: causal self-attention, the transformer block, and the full GPT.

Shapes: B = batch, T = sequence length, C = n_embd, H = n_head, D = C // H (head size).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.embeddings import GPTEmbedding


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    block_size: int = 128     # how many tokens of context the model sees
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 288
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where position t may only look at positions <= t.

    attention(Q, K, V) = softmax(Q K^T / sqrt(D) + mask) V
    The mask puts -inf above the diagonal, so after softmax the future gets weight 0.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        if cfg.n_embd % cfg.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)   # Q, K and V in one matmul
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)      # mix the heads back together
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size), persistent=False)
        self.last_attention: torch.Tensor | None = None    # kept for inspection / tests

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)
        # (B, T, C) -> (B, H, T, D)
        q, k, v = (z.view(b, t, self.n_head, c // self.n_head).transpose(1, 2) for z in (q, k, v))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))          # (B, H, T, T)
        scores = scores.masked_fill(~self.mask[:, :, :t, :t], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        self.last_attention = weights.detach()
        y = self.attn_drop(weights) @ v                                      # (B, H, T, D)
        y = y.transpose(1, 2).contiguous().view(b, t, c)                     # re-assemble heads
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd), nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(ln(x)), then x + mlp(ln(x))."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = GPTEmbedding(cfg.vocab_size, cfg.block_size, cfg.n_embd, cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.head.weight = self.embed.token.weight          # weight tying
        for name, p in self.named_parameters():             # GPT-2 style scaled init for residual outputs
            if name.endswith("proj.weight") or name.endswith("net.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx (B, T) -> logits (B, T, vocab). With targets, also returns the cross-entropy loss."""
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    # ------------------------------------------------------------------ checkpoints
    def save(self, path, extra: dict | None = None) -> None:
        torch.save({"config": asdict(self.cfg), "state_dict": self.state_dict(), **(extra or {})}, path)

    @classmethod
    def load(cls, path, device: str = "cpu") -> tuple["GPT", dict]:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(GPTConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["state_dict"])
        return model.to(device).eval(), ckpt
