"""Milestone 1: build a small corpus of public domain adventure / fantasy stories.

Books come from Project Gutenberg (all public domain in the US). We download the plain-text
editions, cut away the Gutenberg licence header/footer and front matter, normalise whitespace
and punctuation, and join everything into one training file: data/corpus.txt.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import httpx

# (Gutenberg id, title, author) — classic adventure & fantasy novels.
BOOKS: list[tuple[int, str, str]] = [
    (120, "Treasure Island", "Robert Louis Stevenson"),
    (421, "Kidnapped", "Robert Louis Stevenson"),
    (2166, "King Solomon's Mines", "H. Rider Haggard"),
    (711, "Allan Quatermain", "H. Rider Haggard"),
    (62, "A Princess of Mars", "Edgar Rice Burroughs"),
    (64, "The Gods of Mars", "Edgar Rice Burroughs"),
    (139, "The Lost World", "Arthur Conan Doyle"),
    (18857, "A Journey to the Centre of the Earth", "Jules Verne"),
]

MIRRORS = [
    "https://gutenberg.pglaf.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
]

START_RE = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK[^*]*\*\*\*", re.I)
END_RE = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK", re.I)

# Characters we keep after cleaning. Anything else is dropped. Keeping the alphabet small
# makes the byte-level BPE vocabulary go further on a small corpus.
_ALLOWED = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n.,;:!?'\"-()")
_REPLACE = {
    "‘": "'", "’": "'", "“": '"', "”": '"', "—": " - ", "–": "-",
    "…": "...", "_": "", "*": "", "\r": "",
}


def download_book(book_id: int, timeout: float = 60.0) -> str:
    """Fetch a Gutenberg plain-text book, trying each mirror in turn."""
    last_error: Exception | None = None
    for template in MIRRORS:
        try:
            r = httpx.get(template.format(id=book_id), timeout=timeout, follow_redirects=True,
                          headers={"User-Agent": "adventure-story-weaver/1.0 (educational project)"})
            r.raise_for_status()
            return r.content.decode("utf-8-sig", errors="replace")
        except httpx.HTTPError as e:  # try the next mirror
            last_error = e
    raise RuntimeError(f"could not download Gutenberg book {book_id}: {last_error}")


def strip_gutenberg(text: str) -> str:
    """Keep only the text between the *** START *** and *** END *** markers."""
    start = START_RE.search(text)
    if start:
        text = text[start.end():]
    end = END_RE.search(text)
    if end:
        text = text[:end.start()]
    return text


def drop_front_matter(text: str) -> str:
    """Skip title page / table of contents: start at the second 'CHAPTER I' if a contents
    list exists, else the first one."""
    hits = [m.start() for m in re.finditer(r"^\s*CHAPTER\s+(I|1|ONE)\b\.?", text, re.M | re.I)]
    if len(hits) >= 2:
        return text[hits[1]:]
    if hits:
        return text[hits[0]:]
    return text


def clean_text(text: str) -> str:
    """Normalise unicode, quotes and whitespace; drop illustration tags and odd characters."""
    text = unicodedata.normalize("NFKD", text)
    for a, b in _REPLACE.items():
        text = text.replace(a, b)
    text = re.sub(r"\[(Illustration|Footnote)[^\]]*\]", "", text, flags=re.I | re.S)
    text = "".join(ch for ch in text if ch in _ALLOWED)
    # Gutenberg wraps lines at ~70 chars: re-join lines inside a paragraph.
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in paragraphs]
    paragraphs = [p for p in paragraphs if keep_paragraph(p)]
    return "\n\n".join(paragraphs) + "\n"


def keep_paragraph(p: str) -> bool:
    """Reject headings, tables of contents and other non-story paragraphs."""
    if not p or ". . ." in p or "gutenberg" in p.lower():
        return False
    letters = [c for c in p if c.isalpha()]
    if len(letters) < 3:
        return False
    upper = sum(c.isupper() for c in letters) / len(letters)
    if upper > 0.5:  # "CHAPTER IV. THE SEA-CHEST", "TREASURE ISLAND"
        return False
    if re.fullmatch(r"(by|chapter|part|book|contents|illustrated)\b.{0,60}", p, re.I):
        return False
    return True


def prepare_book(raw: str) -> str:
    return clean_text(drop_front_matter(strip_gutenberg(raw)))


def build_corpus(out_path: str | Path = "data/corpus.txt", raw_dir: str | Path = "data/raw",
                 books: list[tuple[int, str, str]] | None = None) -> dict:
    """Download (cached in raw_dir), clean and join the books. Returns simple statistics."""
    books = books or BOOKS
    raw_dir, out_path = Path(raw_dir), Path(out_path)
    raw_dir.mkdir(parents=True, exist_ok=True)
    pieces, stats = [], {"books": []}
    for book_id, title, author in books:
        cache = raw_dir / f"pg{book_id}.txt"
        if not cache.exists():
            print(f"downloading {title} ({book_id})")
            cache.write_text(download_book(book_id))
        cleaned = prepare_book(cache.read_text())
        pieces.append(cleaned)
        stats["books"].append({"id": book_id, "title": title, "author": author, "chars": len(cleaned)})
    corpus = "\n".join(pieces)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(corpus)
    stats["chars"] = len(corpus)
    stats["words"] = len(corpus.split())
    return stats


def load_corpus(path: str | Path = "data/corpus.txt") -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `python -m app.cli corpus` first")
    return path.read_text()
