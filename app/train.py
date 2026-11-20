"""Milestone 4: train the GPT on the adventure corpus.

Recipe: next-token prediction with cross-entropy, AdamW, linear warmup then cosine decay of the
learning rate, gradient clipping, a held-out validation split (last 10% of the corpus), and the
checkpoint with the best validation loss is kept.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from app.corpus import load_corpus
from app.embeddings import encode_corpus
from app.model import GPT, GPTConfig
from app.tokenizer import BPETokenizer


@dataclass
class TrainConfig:
    corpus_path: str = "data/corpus.txt"
    out_dir: str = "checkpoints"
    vocab_size: int = 4096
    model: GPTConfig = field(default_factory=GPTConfig)
    batch_size: int = 32
    max_steps: int = 5000
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_batches: int = 40
    val_fraction: float = 0.1
    seed: int = 1337
    device: str = "auto"


def pick_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def split_data(ids: torch.Tensor, val_fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(len(ids) * (1 - val_fraction))
    return ids[:n], ids[n:]


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str = "cpu",
              generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Random windows of length block_size; targets are the same windows shifted by one token."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model: GPT, data: torch.Tensor, cfg: TrainConfig, device: str) -> float:
    model.eval()
    g = torch.Generator().manual_seed(0)  # same batches every time -> comparable numbers
    losses = [model(*get_batch(data, model.cfg.block_size, cfg.batch_size, device, g))[1].item()
              for _ in range(cfg.eval_batches)]
    model.train()
    return sum(losses) / len(losses)


def make_optimizer(model: GPT, cfg: TrainConfig) -> torch.optim.AdamW:
    """Weight decay on matrices only (not on biases, LayerNorm or embeddings' 1-D params)."""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr, betas=(0.9, 0.95))


def train(cfg: TrainConfig, text: str | None = None, tokenizer: BPETokenizer | None = None,
          log=print) -> dict:
    """Train a model and save tokenizer.json, model.pt and train_log.json into cfg.out_dir."""
    torch.manual_seed(cfg.seed)
    device = pick_device(cfg.device)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    text = text if text is not None else load_corpus(cfg.corpus_path)
    if tokenizer is None:
        tok_path = out / "tokenizer.json"
        if tok_path.exists():
            tokenizer = BPETokenizer.load(tok_path)
        else:
            log(f"training BPE tokenizer (vocab {cfg.vocab_size})")
            tokenizer = BPETokenizer().train(text, cfg.vocab_size)
            tokenizer.save(tok_path)
    ids = encode_corpus(tokenizer, text)
    train_data, val_data = split_data(ids, cfg.val_fraction)

    cfg.model.vocab_size = tokenizer.vocab_size
    model = GPT(cfg.model).to(device)
    opt = make_optimizer(model, cfg)
    log(f"device={device} params={model.num_params() / 1e6:.2f}M tokens={len(ids):,} "
        f"(train {len(train_data):,} / val {len(val_data):,})")

    history, best_val, start = [], float("inf"), time.time()
    g = torch.Generator().manual_seed(cfg.seed)
    for step in range(cfg.max_steps + 1):
        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            tr = estimate_loss(model, train_data, cfg, device)
            va = estimate_loss(model, val_data, cfg, device)
            history.append({"step": step, "train_loss": round(tr, 4), "val_loss": round(va, 4),
                            "lr": lr_at(step, cfg), "elapsed_s": round(time.time() - start, 1)})
            log(f"step {step:5d} | train {tr:.3f} | val {va:.3f} | ppl {math.exp(va):7.1f} "
                f"| {time.time() - start:6.0f}s")
            if va < best_val:
                best_val = va
                model.save(out / "model.pt", {"step": step, "val_loss": va, "train_config": _cfg_dict(cfg)})
        if step == cfg.max_steps:
            break
        for group in opt.param_groups:
            group["lr"] = lr_at(step, cfg)
        x, y = get_batch(train_data, cfg.model.block_size, cfg.batch_size, device, g)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

    result = {"best_val_loss": best_val, "best_val_perplexity": math.exp(best_val),
              "params": model.num_params(), "tokens": len(ids), "device": device,
              "train_seconds": round(time.time() - start, 1), "config": _cfg_dict(cfg), "history": history}
    (out / "train_log.json").write_text(json.dumps(result, indent=2))
    return result


def _cfg_dict(cfg: TrainConfig) -> dict:
    d = asdict(cfg)
    return d
