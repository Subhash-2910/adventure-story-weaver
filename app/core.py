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
