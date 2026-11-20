"""The milestones in one place. Each function is a small, documented entry point into the module
that implements that milestone, so you can follow the pipeline from raw text to generated story.
"""
from __future__ import annotations

from pathlib import Path


# --- Milestone 1: Assemble a small corpus of text from public domain adventure stories or fantasy novels.
def milestone_1(out_path: str | Path = "data/corpus.txt") -> dict:
    """Download, clean and join the public domain books listed in app/corpus.py."""
    from app.corpus import build_corpus
    return build_corpus(out_path)


# --- Milestone 2: Implement the byte-pair tokenizer and generate embeddings from the story text.
def milestone_2(text: str, vocab_size: int = 4096, n_embd: int = 64, block_size: int = 64):
    """Train a BPE tokenizer on `text`, then embed the first block of tokens.

    Returns (tokenizer, ids, vectors) where vectors has shape (1, T, n_embd)."""
    import torch

    from app.embeddings import GPTEmbedding
    from app.tokenizer import BPETokenizer
    tok = BPETokenizer().train(text, vocab_size)
    ids = tok.encode(text)[:block_size]
    emb = GPTEmbedding(tok.vocab_size, block_size, n_embd)
    return tok, ids, emb(torch.tensor([ids]))


# --- Milestone 3: Build the causal self-attention mechanism and assemble the transformer block.
def milestone_3(vocab_size: int = 4096, **overrides):
    """Build a GPT (embeddings -> N x [causal self-attention + MLP] blocks -> logits)."""
    from app.model import GPT, GPTConfig
    return GPT(GPTConfig(vocab_size=vocab_size, **overrides))
