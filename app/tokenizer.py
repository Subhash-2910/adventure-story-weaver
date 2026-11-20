"""Milestone 2a: a byte-level Byte-Pair Encoding (BPE) tokenizer, written from scratch.

How it works
1. Text is split into chunks (words with their leading space, numbers, punctuation, whitespace)
   so merges never cross word boundaries - the same idea as GPT-2.
2. Every chunk starts as a list of UTF-8 bytes, so tokens 0..255 are the 256 single bytes and
   ANY text can be encoded (no unknown token).
3. Training repeatedly finds the most frequent adjacent pair of tokens and merges it into a new
   token, until the vocabulary reaches `vocab_size`.
4. Encoding applies the learned merges in the order they were learned; decoding just joins bytes.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SPLIT_RE = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")


def split_chunks(text: str) -> list[str]:
    return SPLIT_RE.findall(text)


class BPETokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges: list[tuple[int, int]] = [tuple(m) for m in (merges or [])]
        self._rebuild()

    def _rebuild(self) -> None:
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    # ------------------------------------------------------------------ training
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> "BPETokenizer":
        """Learn `vocab_size - 256` merges from `text`.

        Works on the unique chunks weighted by frequency, and keeps pair counts up to date
        incrementally, so training a few thousand merges on a few MB of text takes seconds.
        """
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 (the byte alphabet)")
        chunk_counts = Counter(split_chunks(text))
        words = [list(c.encode("utf-8")) for c in chunk_counts]
        freqs = list(chunk_counts.values())

        pair_counts: Counter = Counter()
        where: dict[tuple[int, int], set[int]] = defaultdict(set)  # pair -> indices of words containing it
        for wi, w in enumerate(words):
            for pair in zip(w, w[1:]):
                pair_counts[pair] += freqs[wi]
                where[pair].add(wi)

        self.merges = []
        for new_id in range(256, vocab_size):
            if not pair_counts:
                break
            best = max(pair_counts, key=pair_counts.get)
            if pair_counts[best] < 2:
                break
            self.merges.append(best)
            for wi in list(where[best]):
                w, f = words[wi], freqs[wi]
                for pair in zip(w, w[1:]):  # remove this word's old pairs
                    pair_counts[pair] -= f
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                w = _merge(w, best, new_id)
                words[wi] = w
                for pair in zip(w, w[1:]):  # add its new pairs
                    pair_counts[pair] += f
                    where[pair].add(wi)
            del where[best]
            if verbose and (new_id - 255) % 500 == 0:
                print(f"  merge {new_id - 255}/{vocab_size - 256}")
        self._rebuild()
        return self

    # ------------------------------------------------------------------ encode / decode
    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # merge the pair that was learned earliest
            pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, float("inf")))
            if pair not in self.ranks:
                break
            ids = _merge(ids, pair, 256 + self.ranks[pair])
        if len(self._cache) < 200_000:
            self._cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for chunk in split_chunks(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def token_strings(self, ids: list[int]) -> list[str]:
        return [self.vocab[i].decode("utf-8", errors="replace") for i in ids]

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"type": "byte-bpe", "merges": self.merges}))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text())
        return cls([tuple(m) for m in data["merges"]])


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i, a, b = [], 0, pair[0], pair[1]
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out
