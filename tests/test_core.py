"""Tests for each milestone. They use tiny inputs so the whole suite runs in seconds on CPU."""
from pathlib import Path

import pytest

from app import corpus

ROOT = Path(__file__).resolve().parent.parent

RAW_BOOK = """The Project Gutenberg eBook of Test Island
Some licence text.
*** START OF THE PROJECT GUTENBERG EBOOK TEST ISLAND ***

TEST ISLAND

CONTENTS

CHAPTER I. THE SHORE . . . . 1

CHAPTER I

The ship came to the shore at dawn,
and the captain said “Land ho!”

[Illustration: a ship]

We walked into the _dark_ jungle—slowly.

*** END OF THE PROJECT GUTENBERG EBOOK TEST ISLAND ***
More licence text about Project Gutenberg.
"""


# ---------------------------------------------------------------- Milestone 1: corpus
def test_milestone_1_cleaning_strips_licence_headings_and_odd_characters():
    text = corpus.prepare_book(RAW_BOOK)
    assert "licence" not in text and "Gutenberg" not in text
    assert "CONTENTS" not in text and "CHAPTER" not in text and "Illustration" not in text
    # wrapped lines are rejoined into one paragraph, curly quotes become plain ones
    assert 'The ship came to the shore at dawn, and the captain said "Land ho!"' in text
    assert "We walked into the dark jungle - slowly." in text
    assert set(text) <= corpus._ALLOWED


def test_milestone_1_build_corpus_uses_cached_raw_books(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "pg1.txt").write_text(RAW_BOOK)
    stats = corpus.build_corpus(tmp_path / "corpus.txt", raw, books=[(1, "Test Island", "Nobody")])
    assert stats["books"][0]["title"] == "Test Island"
    assert (tmp_path / "corpus.txt").read_text().startswith("The ship came")
    assert stats["words"] > 10


def test_milestone_1_committed_corpus_is_real_and_large():
    text = (ROOT / "data" / "corpus.txt").read_text()
    assert len(text.split()) > 500_000
    assert "Treasure" in text and "Mars" in text


# Milestone 2: Implement the byte-pair tokenizer and generate embeddings from the story text.
@pytest.mark.skip(reason="milestone 2 not started")
def test_milestone_2():
    raise NotImplementedError


# Milestone 3: Build the causal self-attention mechanism and assemble the transformer block.
@pytest.mark.skip(reason="milestone 3 not started")
def test_milestone_3():
    raise NotImplementedError


# Milestone 4: Set up the training loop and train the GPT model on the narrative data.
@pytest.mark.skip(reason="milestone 4 not started")
def test_milestone_4():
    raise NotImplementedError


# Milestone 5: Implement a text generation feature that creates coherent story continuations.
@pytest.mark.skip(reason="milestone 5 not started")
def test_milestone_5():
    raise NotImplementedError


# Milestone 6: Design a CLI where users can input a story prompt and receive generated segments.
@pytest.mark.skip(reason="milestone 6 not started")
def test_milestone_6():
    raise NotImplementedError
