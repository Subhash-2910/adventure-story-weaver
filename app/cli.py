"""Command-line entry point: `python -m app.cli --help`.

Milestone 6: a CLI where you type a story prompt and get generated segments back.

  python -m app.cli generate "The cave mouth was black" --length 150 --temperature 0.8
  python -m app.cli play                       # interactive: write, generate, continue
  python -m app.cli story new cave "The cave mouth was black"   # saved, continue later
  python -m app.cli story continue cave
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = ROOT / "checkpoints"
STORIES_PATH = ROOT / "stories" / "stories.json"


# ---------------------------------------------------------------- helpers
def _wrap(text: str, width: int = 88) -> str:
    return "\n\n".join(textwrap.fill(p.strip(), width) for p in text.split("\n\n") if p.strip())


def _sampling(args):
    from app.generate import SamplingConfig
    return SamplingConfig(max_new_tokens=args.length, temperature=args.temperature, top_k=args.top_k or None,
                          top_p=args.top_p if 0 < args.top_p < 1 else None,
                          repetition_penalty=args.repetition_penalty, seed=args.seed,
                          stop_at_sentence=not args.no_stop)


def _generator(args):
    from app.generate import StoryGenerator
    return StoryGenerator(args.checkpoints)


def _store():
    from app.storage import Store
    return Store(str(STORIES_PATH))


def add_sampling_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("generation settings")
    g.add_argument("--length", type=int, default=120, help="tokens to generate (~0.75 words each, default 120)")
    g.add_argument("--temperature", "--creativity", type=float, default=0.8,
                   help="creativity: 0.3 safe/repetitive, 0.8 balanced, 1.2 wild (default 0.8)")
    g.add_argument("--top-k", type=int, default=50, help="sample only from the k likeliest tokens (0 = off)")
    g.add_argument("--top-p", type=float, default=0.95, help="nucleus sampling threshold (1 = off)")
    g.add_argument("--repetition-penalty", type=float, default=1.15, help="discourage repeating recent tokens")
    g.add_argument("--seed", type=int, default=None, help="fix the random seed for reproducible output")
    g.add_argument("--no-stop", action="store_true", help="don't extend to finish the last sentence")


# ---------------------------------------------------------------- commands
def cmd_hello(args) -> None:
    print("Adventure Story Weaver is ready. Open README.md for your milestones.")
    model = Path(args.checkpoints) / "model.pt"
    print(f"Trained model: {'found' if model.exists() else 'missing - run `python -m app.cli train`'} ({model})")


def cmd_corpus(args) -> None:
    from app.corpus import build_corpus
    stats = build_corpus(args.out)
    for b in stats["books"]:
        print(f"  {b['title']:<40} {b['author']:<24} {b['chars']:>9,} chars")
    print(f"corpus: {stats['words']:,} words, {stats['chars']:,} chars -> {args.out}")


def cmd_tokenize(args) -> None:
    from app.tokenizer import BPETokenizer
    tok = BPETokenizer.load(Path(args.checkpoints) / "tokenizer.json")
    ids = tok.encode(args.text)
    print(f"{len(ids)} tokens for {len(args.text)} characters (vocab {tok.vocab_size})")
    print(" | ".join(repr(s) for s in tok.token_strings(ids)))
    print(ids)


def cmd_train(args) -> None:
    from app.model import GPTConfig
    from app.train import TrainConfig, train
    model = GPTConfig(n_layer=args.layers, n_head=args.heads, n_embd=args.embd, block_size=args.block,
                      dropout=args.dropout)
    cfg = TrainConfig(out_dir=args.checkpoints, max_steps=args.steps, batch_size=args.batch, lr=args.lr,
                      vocab_size=args.vocab, model=model, device=args.device,
                      eval_every=max(1, min(250, args.steps // 4)))
    if args.quick:
        cfg.model = GPTConfig(n_layer=2, n_head=4, n_embd=128, block_size=64, dropout=0.1)
        cfg.max_steps, cfg.eval_every = 300, 100
    log = train(cfg)
    print(f"best validation loss {log['best_val_loss']:.3f} (perplexity {log['best_val_perplexity']:.1f}) "
          f"in {log['train_seconds']:.0f}s -> {args.checkpoints}/model.pt")


def cmd_generate(args) -> None:
    prompt = args.prompt if args.prompt is not None else sys.stdin.read()
    gen = _generator(args)
    cfg = _sampling(args)
    for i in range(args.num):
        if args.num > 1:
            print(f"--- version {i + 1} ---")
            if cfg.seed is not None:
                cfg.seed += i
        text = gen.continue_story(prompt, cfg)
        print(_wrap(prompt + "→" + text if args.mark else prompt + text))
        print()


def cmd_play(args) -> None:
    gen = _generator(args)
    cfg = _sampling(args)
    print("Adventure Story Weaver - interactive mode")
    print("Type the start of a scene and press Enter. Commands: /more  /temp 0.9  /length 200  "
          "/save NAME  /new  /show  /quit\n")
    story = ""
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/quit", "/exit", "/q"):
            break
        if line == "/new":
            story = ""
            print("(new story)\n")
            continue
        if line == "/show":
            print(_wrap(story) or "(empty)", "\n")
            continue
        if line.startswith("/temp"):
            cfg.temperature = float(line.split()[1])
            print(f"(temperature {cfg.temperature})")
            continue
        if line.startswith("/length"):
            cfg.max_new_tokens = int(line.split()[1])
            print(f"(length {cfg.max_new_tokens} tokens)")
            continue
        if line.startswith("/save"):
            name = line.split(maxsplit=1)[1] if " " in line else "untitled"
            _store().put(name, {"text": story})
            print(f"(saved as '{name}' in {STORIES_PATH})")
            continue
        if line != "/more":
            story = (story + " " + line).strip() if story else line
        segment = gen.continue_story(story, cfg)
        story += segment
        print("\nweaver> " + _wrap(segment.strip()) + "\n")


def cmd_story(args) -> None:
    store = _store()
    if args.action == "list":
        stories = store.all()
        if not stories:
            print("no saved stories yet: python -m app.cli story new NAME \"your opening line\"")
        for name, s in stories.items():
            print(f"  {name:<20} {len(s['text'].split()):>5} words  {s['text'][:60]!r}...")
        return
    if not args.name:
        raise SystemExit("give a story name")
    if args.action == "show":
        s = store.get(args.name)
        if s is None:
            raise SystemExit(f"no story called {args.name!r}")
        print(_wrap(s["text"]))
        return
    if args.action == "delete":
        data = store.all()
        data.pop(args.name, None)
        store.path.write_text(json.dumps(data, indent=2))
        print(f"deleted {args.name}")
        return
    if args.action == "export":
        s = store.get(args.name)
        if s is None:
            raise SystemExit(f"no story called {args.name!r}")
        out = Path(args.text or f"{args.name}.txt")
        out.write_text(_wrap(s["text"]) + "\n")
        print(f"wrote {out}")
        return
    gen = _generator(args)
    cfg = _sampling(args)
    if args.action == "new":
        if not args.text:
            raise SystemExit("give an opening line: story new NAME \"The cave mouth was black\"")
        text = args.text
    else:  # continue
        s = store.get(args.name)
        if s is None:
            raise SystemExit(f"no story called {args.name!r}: start it with `story new`")
        text = s["text"] + (" " + args.text if args.text else "")
    segment = gen.continue_story(text, cfg)
    store.put(args.name, {"text": text + segment})
    print(_wrap(segment.strip()))
    print(f"\n(saved '{args.name}': {len((text + segment).split())} words)")


def cmd_evaluate(args) -> None:
    from app.evaluate import evaluate
    report = evaluate(args.checkpoints, args.out)
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))
    print(f"full report -> {args.out}")


# ---------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adventure-story-weaver",
        description="A tiny GPT, trained from scratch on public domain adventure novels, "
                    "that continues your story prompts.")
    parser.add_argument("--checkpoints", default=str(CHECKPOINTS), help="folder with tokenizer.json and model.pt")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("hello", help="check that everything is wired up")

    p = sub.add_parser("corpus", help="download and clean the public domain books (milestone 1)")
    p.add_argument("--out", default=str(ROOT / "data" / "corpus.txt"))

    p = sub.add_parser("tokenize", help="show how the BPE tokenizer splits some text (milestone 2)")
    p.add_argument("text")

    p = sub.add_parser("train", help="train the GPT on the corpus (milestone 4)")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--vocab", type=int, default=4096)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--embd", type=int, default=288)
    p.add_argument("--block", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--device", default="auto", help="auto, cpu, mps or cuda")
    p.add_argument("--quick", action="store_true", help="tiny 2-minute model, to check the pipeline")

    p = sub.add_parser("generate", help="continue a story prompt (milestones 5 + 6)")
    p.add_argument("prompt", nargs="?", help="the start of your story (or pipe it on stdin)")
    p.add_argument("-n", "--num", type=int, default=1, help="how many alternative continuations")
    p.add_argument("--mark", action="store_true", help="show where the prompt ends with an arrow")
    add_sampling_args(p)

    p = sub.add_parser("play", help="interactive story mode")
    add_sampling_args(p)

    p = sub.add_parser("story", help="save, continue, show, list, export or delete longer stories")
    p.add_argument("action", choices=["new", "continue", "show", "list", "export", "delete"])
    p.add_argument("name", nargs="?")
    p.add_argument("text", nargs="?", help="opening line (new), extra line to add (continue), or file (export)")
    add_sampling_args(p)

    p = sub.add_parser("evaluate", help="compare the GPT with a bigram baseline and write sample outputs")
    p.add_argument("--out", default=str(ROOT / "results" / "metrics.json"))
    return parser


COMMANDS = {"hello": cmd_hello, "corpus": cmd_corpus, "tokenize": cmd_tokenize, "train": cmd_train,
            "generate": cmd_generate, "play": cmd_play, "story": cmd_story, "evaluate": cmd_evaluate}


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
