"""Tests for each milestone. They use tiny inputs so the whole suite runs in seconds on CPU."""
from pathlib import Path

import pytest
import torch

from app import core, corpus
from app.embeddings import GPTEmbedding
from app.model import GPT, Block, CausalSelfAttention, GPTConfig
from app.tokenizer import BPETokenizer

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


# ---------------------------------------------------------------- Milestone 2: tokenizer + embeddings
STORY = ("The captain drew his sword. The captain looked at the dark sea, and the crew looked at the captain. "
         "\"Treasure!\" cried the boy. The treasure lay deep in the cave. ") * 20


def test_milestone_2_bpe_learns_merges_and_round_trips():
    tok = BPETokenizer().train(STORY, vocab_size=300)
    assert tok.vocab_size == 300
    ids = tok.encode(STORY)
    assert len(ids) < len(STORY.encode()) / 2          # merges compress the text
    assert tok.decode(ids) == STORY                     # lossless
    assert " captain" in tok.token_strings(tok.encode(" captain"))  # frequent word became one token


def test_milestone_2_bpe_handles_unseen_text_and_save_load(tmp_path):
    tok = BPETokenizer().train(STORY, vocab_size=280)
    unseen = "Zebras \u00e9t\u00e9 42 xylophone!"
    assert tok.decode(tok.encode(unseen)) == unseen      # byte fallback: nothing is unknown
    tok.save(tmp_path / "tok.json")
    again = BPETokenizer.load(tmp_path / "tok.json")
    assert again.encode(STORY) == tok.encode(STORY)


def test_milestone_2_embeddings_shape_and_position_matters():
    tok, ids, vectors = core.milestone_2(STORY, vocab_size=300, n_embd=16, block_size=32)
    assert vectors.shape == (1, 32, 16)
    emb = GPTEmbedding(tok.vocab_size, block_size=8, n_embd=16)
    same = torch.tensor([[5, 5]])
    out = emb(same)
    assert not torch.allclose(out[0, 0], out[0, 1])     # same token, different position -> different vector
    with pytest.raises(ValueError):
        emb(torch.zeros(1, 9, dtype=torch.long))


# ---------------------------------------------------------------- Milestone 3: attention + transformer
TINY = dict(vocab_size=50, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)


def test_milestone_3_attention_is_causal():
    torch.manual_seed(0)
    cfg = GPTConfig(**TINY)
    attn = CausalSelfAttention(cfg).eval()
    x = torch.randn(1, 10, 32)
    y = attn(x)
    w = attn.last_attention[0]                          # (H, T, T)
    assert torch.all(w.triu(diagonal=1) == 0)           # nothing attends to the future
    assert torch.allclose(w.sum(-1), torch.ones(2, 10))  # each row is a probability distribution
    # changing the last token must not change earlier outputs
    x2 = x.clone()
    x2[0, -1] += 5.0
    assert torch.allclose(attn(x2)[0, :-1], y[0, :-1], atol=1e-6)


def test_milestone_3_block_and_gpt_shapes():
    torch.manual_seed(0)
    cfg = GPTConfig(**TINY)
    x = torch.randn(3, 7, 32)
    assert Block(cfg)(x).shape == x.shape
    model = core.milestone_3(**TINY)
    idx = torch.randint(0, 50, (3, 16))
    logits, loss = model(idx, idx)
    assert logits.shape == (3, 16, 50)
    # an untrained model is close to uniform: loss ~ ln(vocab)
    assert abs(loss.item() - torch.log(torch.tensor(50.0)).item()) < 0.5
    assert model.head.weight is model.embed.token.weight  # weight tying


def test_milestone_3_checkpoint_round_trip(tmp_path):
    model = GPT(GPTConfig(**TINY)).eval()
    model.save(tmp_path / "m.pt", {"note": "hi"})
    again, ckpt = GPT.load(tmp_path / "m.pt")
    idx = torch.randint(0, 50, (1, 8))
    assert torch.allclose(model(idx)[0], again(idx)[0]) and ckpt["note"] == "hi"


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
