"""D1 Task 1. The verse corpus as (meta, text) pairs with a train/val split.

Source 1: a JSONL file with fields author, title, text, split (C2's verse corpus).
Source 2 (fallback when the file does not exist): line windows of the Gutenberg books in data/corpus.txt.
Each window of n consecutive non-empty lines is one "poem". The header keeps the book's author and title.
The split is by cluster: the last `val_frac` of each book is validation, so no window crosses the split.
"""
import json
import re
from pathlib import Path

import numpy as np

# Tolerant markers: "***END OF..." without a space, and old "End of the Project Gutenberg EBook" lines.
# The strict form missed 4 of 40 verse books, and their licence text leaked into the corpus.
_START = re.compile(r"^\W*START OF TH(?:E|IS) PROJECT GUTENBERG E-?BOOK.*$", re.I | re.M)
_END = re.compile(r"^\W*END OF TH(?:E|IS) PROJECT GUTENBERG E-?BOOK.*$", re.I | re.M)
_HEADING = re.compile(r"^(CHAPTER|BOOK|PART|VOLUME|EPILOGUE|PREFACE)\b|^[IVXLC]+\.?$")

Poem = tuple[dict, str]


def load_jsonl(path: Path) -> dict[str, list[Poem]]:
    out: dict[str, list[Poem]] = {"train": [], "val": []}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out["val" if r.get("split") in ("val", "validation", "test") else "train"].append(
            ({"author": r["author"], "title": r["title"]}, r["text"]))
    return out


def gutenberg_books(text: str) -> list[tuple[dict, list[str]]]:
    books = []
    for m in _START.finditer(text):
        head = text[max(0, m.start() - 3000) : m.start()]
        get = lambda k: (re.findall(rf"^{k}: (.+)$", head, re.M) or ["Unknown"])[-1].strip()
        end = _END.search(text, m.end())
        body = text[m.end() : end.start() if end else len(text)]
        lines = [l.strip() for l in body.splitlines()]
        lines = [l for l in lines if l and not _HEADING.match(l) and not l.startswith("[Illustration")]
        books.append(({"author": get("Author"), "title": get("Title")}, lines))
    return books


def prose_windows(path: Path, n_min: int = 4, n_max: int = 24, val_frac: float = 0.1,
                  seed: int = 0) -> dict[str, list[Poem]]:
    rng = np.random.default_rng(seed)
    out: dict[str, list[Poem]] = {"train": [], "val": []}
    for meta, lines in gutenberg_books(path.read_text(encoding="utf-8")):
        cut = int(len(lines) * (1 - val_frac))
        for split, (lo, hi) in (("train", (0, cut)), ("val", (cut, len(lines)))):
            i, k = lo, 0
            while i < hi:
                n = int(rng.integers(n_min, n_max + 1))
                if i + n > hi:
                    break
                k += 1
                out[split].append(({"author": meta["author"], "title": f"{meta['title']} {k}"},
                                   "\n".join(lines[i : i + n])))
                i += n
    return out


def load_verse(path: str = "data/verse.jsonl", fallback: str = "data/corpus.txt") -> dict[str, list[Poem]]:
    p = Path(path)
    return load_jsonl(p) if p.exists() else prose_windows(Path(fallback))


# ---------------------------------------------------------------- C2: anthologies into poems

_POEM_BREAK = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")  # two or more empty lines
_ATTRIBUTION = re.compile(r"^(?P<name>[A-Z][\w.'\- ]+?)\s*\[(?P<dates>[^\]]*\d[^\]]*)\]$")


def gutenberg_body(text: str) -> str:
    """The text between the START and END markers. The whole text if a marker is missing."""
    m = _START.search(text)
    body = text[m.end():] if m else text
    e = _END.search(body)
    return body[: e.start()] if e else body


def _is_title(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    return bool(letters) and len(line.split()) <= 12 and sum(c.isupper() for c in letters) / len(letters) > 0.8


def split_anthology(body: str, default_author: str) -> list[dict]:
    """Poems from a Gutenberg verse book. A poem is a block between two or more empty lines.

    An all-caps first line is the title (a "From ..." line after it is the source). A last line of the form
    "Name [dates]" is the author (anthologies). A block with only a title gives its title to the next block.
    """
    poems, pending = [], None
    for block in _POEM_BREAK.split(body.replace("\r\n", "\n")):
        lines = [l.rstrip() for l in block.strip("\n").split("\n")]
        if not any(lines):
            continue
        title, author = None, default_author
        if _is_title(lines[0].strip()):
            title, lines = lines[0].strip(), lines[1:]
            if lines and lines[0].strip().startswith("From "):
                lines = lines[1:]
        m = _ATTRIBUTION.match(lines[-1].strip()) if lines else None
        if m:
            author, lines = m["name"].strip(), lines[:-1]
        if title and not any(l.strip() for l in lines):
            pending = title
            continue
        title = title or pending
        pending = None
        poems.append({"title": title or "Untitled", "author": author, "text": "\n".join(lines).strip("\n")})
    return poems
