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


# --- Milestone 4: Set up the training loop and train the GPT model on the narrative data.
def milestone_4(**overrides) -> dict:
    """Train the GPT (see app/train.py for the recipe). Returns the training log with losses."""
    from app.train import TrainConfig, train
    return train(TrainConfig(**overrides))


# --- Milestone 5: Implement a text generation feature that creates coherent story continuations.
def milestone_5(prompt: str, checkpoint_dir: str | Path | None = None, **sampling) -> str:
    """Continue `prompt` with the trained model using temperature / top-k / top-p sampling."""
    from app.generate import DEFAULT_DIR, SamplingConfig, StoryGenerator
    gen = StoryGenerator(checkpoint_dir or DEFAULT_DIR)
    return gen.continue_story(prompt, SamplingConfig(**sampling))


# --- Milestone 6: Design a CLI where users can input a story prompt and receive generated segments.
def milestone_6(argv: list[str] | None = None) -> None:
    """Run the command-line interface (see app/cli.py, or `python -m app.cli --help`)."""
    from app.cli import main
    main(argv)
