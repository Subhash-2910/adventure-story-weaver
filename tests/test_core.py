"""Tests for each milestone. They use tiny inputs so the whole suite runs in seconds on CPU."""
import json
from pathlib import Path

import pytest
import torch

from app import cli, core, corpus
from app.baseline import BigramBaseline
from app.embeddings import GPTEmbedding
from app.generate import SamplingConfig, StoryGenerator, apply_repetition_penalty, filter_logits
from app.model import GPT, Block, CausalSelfAttention, GPTConfig
from app.tokenizer import BPETokenizer
from app.train import TrainConfig, get_batch, lr_at, train

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


# ---------------------------------------------------------------- Milestone 4: training loop
def _tiny_train(tmp_path, steps=60):
    tok = BPETokenizer().train(STORY, vocab_size=300)
    cfg = TrainConfig(out_dir=str(tmp_path), model=GPTConfig(**{**TINY, "vocab_size": 300}),
                      batch_size=8, max_steps=steps, eval_every=steps // 2, eval_batches=3,
                      warmup_steps=5, lr=3e-3, device="cpu", val_fraction=0.2)
    return tok, train(cfg, text=STORY, tokenizer=tok, log=lambda *_: None)


def test_milestone_4_training_reduces_loss_and_saves_checkpoint(tmp_path):
    tok, log = _tiny_train(tmp_path)
    first, last = log["history"][0], log["history"][-1]
    assert last["train_loss"] < first["train_loss"] - 1.0     # it actually learns
    assert log["best_val_loss"] < first["val_loss"]
    assert (tmp_path / "model.pt").exists() and (tmp_path / "train_log.json").exists()


def test_milestone_4_batches_are_shifted_by_one_and_lr_schedule():
    data = torch.arange(100)
    x, y = get_batch(data, block_size=10, batch_size=4)
    assert torch.equal(x + 1, y)
    cfg = TrainConfig(lr=1.0, min_lr=0.1, warmup_steps=10, max_steps=110)
    assert lr_at(0, cfg) == pytest.approx(0.1) and lr_at(9, cfg) == pytest.approx(1.0)
    assert lr_at(110, cfg) == pytest.approx(0.1)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """One small model trained for 120 steps, shared by the tests below."""
    path = tmp_path_factory.mktemp("trained")
    tok, log = _tiny_train(path, steps=120)
    tok.save(path / "tokenizer.json")
    model, _ = GPT.load(path / "model.pt")
    return tok, log, model, path


def test_milestone_4_gpt_beats_bigram_baseline_on_learnable_text(trained):
    tok, log, _, _ = trained
    ids = torch.tensor(tok.encode(STORY))
    bigram = BigramBaseline(tok.vocab_size).fit(ids)
    assert log["best_val_loss"] < bigram.loss(ids)


# ---------------------------------------------------------------- Milestone 5: generation
def test_milestone_5_top_k_top_p_and_repetition_penalty():
    logits = torch.tensor([3.0, 2.0, 1.0, 0.0, -1.0])
    assert torch.isinf(filter_logits(logits, top_k=2)[2:]).all()
    kept = filter_logits(logits, top_p=0.7)
    assert not torch.isinf(kept[0]) and torch.isinf(kept[-1])
    pen = apply_repetition_penalty(logits, [0, 4], 2.0)
    assert pen[0] == 1.5 and pen[4] == -2.0 and pen[1] == 2.0


def test_milestone_5_generation_continues_a_prompt(trained):
    tok, _, model, _ = trained
    gen = StoryGenerator.from_parts(model, tok)
    cfg = SamplingConfig(max_new_tokens=15, temperature=0.5, seed=1, stop_at_sentence=False)
    a = gen.continue_story("The captain", cfg)
    assert a and a == gen.continue_story("The captain", cfg)          # seeded -> reproducible
    assert len(tok.encode(a)) == 15
    assert any(w in a for w in ("captain", "the", "Treasure", "cave", "sea"))  # learned the story's words
    with pytest.raises(ValueError):
        gen.continue_story("x", SamplingConfig(temperature=0))


def test_milestone_5_stop_at_sentence_ends_cleanly(trained):
    tok, _, model, _ = trained
    gen = StoryGenerator.from_parts(model, tok)
    out = gen.continue_story("The captain", SamplingConfig(max_new_tokens=5, temperature=0.3, seed=3))
    assert gen._ends_sentence(tok.encode(out)) or len(tok.encode(out)) == 45


# ---------------------------------------------------------------- Milestone 6: command-line interface
def _cli(capsys, *argv, ckpt=None):
    args = (["--checkpoints", str(ckpt)] if ckpt else []) + list(argv)
    cli.main(args)
    return capsys.readouterr().out


def test_milestone_6_generate_command_prints_prompt_and_continuation(trained, capsys):
    out = _cli(capsys, "generate", "The captain", "--length", "12", "--seed", "1", "--temperature", "0.5",
               ckpt=trained[3])
    assert out.startswith("The captain") and len(out.split()) > 4
    two = _cli(capsys, "generate", "The captain", "-n", "2", "--length", "5", "--seed", "1", ckpt=trained[3])
    assert "version 1" in two and "version 2" in two


def test_milestone_6_story_save_continue_show_list(trained, capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "STORIES_PATH", tmp_path / "stories.json")
    _cli(capsys, "story", "new", "cave", "The captain", "--length", "8", "--seed", "2", ckpt=trained[3])
    first = json.loads((tmp_path / "stories.json").read_text())["cave"]["text"]
    assert first.startswith("The captain") and len(first) > len("The captain")
    _cli(capsys, "story", "continue", "cave", "--length", "8", "--seed", "3", ckpt=trained[3])
    second = json.loads((tmp_path / "stories.json").read_text())["cave"]["text"]
    assert second.startswith(first) and len(second) > len(first)        # the story grew
    assert "cave" in _cli(capsys, "story", "list")
    assert "captain" in _cli(capsys, "story", "show", "cave")
    with pytest.raises(SystemExit):
        _cli(capsys, "story", "continue", "missing", ckpt=trained[3])


def test_milestone_6_interactive_play_session(trained, capsys, monkeypatch):
    lines = iter(["The captain", "/temp 0.4", "/more", "/show", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(lines))
    out = _cli(capsys, "play", "--length", "6", "--seed", "4", ckpt=trained[3])
    assert out.count("weaver>") == 2 and "temperature 0.4" in out


def test_milestone_6_tokenize_and_helpful_errors(trained, capsys, tmp_path):
    out = _cli(capsys, "tokenize", "The captain", ckpt=trained[3])
    assert "tokens for 11 characters" in out
    with pytest.raises(FileNotFoundError, match="train first"):
        _cli(capsys, "generate", "hi", ckpt=tmp_path)


# ---------------------------------------------------------------- evaluation helpers
def test_evaluation_copy_check_and_exact_validation_loss(trained):
    from app.evaluate import copied_fraction, full_val_loss
    grams = {"one two three four five six seven eight"}
    assert copied_fraction("one two three four five six seven eight", grams) == 1.0
    assert copied_fraction("one two three four five six seven nine", grams) == 0.0
    tok, _, model, _ = trained
    ids = torch.tensor(tok.encode(STORY))
    loss = full_val_loss(model, ids, model.cfg.block_size)
    assert 0 < loss < BigramBaseline(tok.vocab_size).fit(ids).loss(ids) + 1
