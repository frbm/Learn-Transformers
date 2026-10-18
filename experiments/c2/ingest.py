"""C2 Task 1: ingest public-domain verse from Project Gutenberg, split anthologies into poems, clean, report the funnel.

python -m experiments.c2.ingest                      # data/c2/verse_raw.jsonl and data/c2/verse_clean.jsonl
python -m experiments.c2.ingest --show max_symbols   # 20 documents that one rule removed, for inspection

Every response is cached under data/c2. A second run downloads nothing.
"""
import argparse
import json
import random
import re
import time
import urllib.request
from pathlib import Path

from lm.data.corpus import normalize_text
from lm.verse.corpus import gutenberg_body, split_anthology

DIR = Path("data/c2")
LISTING = "https://gutendex.com/books/?topic=poetry&languages=en&page={}"
# Prose translations, collected works with plays, and book-length epics are not "poems" for D1's headers.
EXCLUDE = re.compile(r"prose|complete works|works of|plays|drama|iliad|odyssey|divine comedy|beowulf|aeneid|"
                     r"paradise lost|faerie queene|canterbury|idylls|letters|life of|essay", re.I)
STOP = set("the and of to a in that is was he for it with as his on be at by i you not this but from or have an "
           "they which one her all she there we my me thy thou thee our your so no what when".split())


def fetch(url: str, path: Path, timeout: int = 120) -> bytes:
    if path.exists():
        return path.read_bytes()
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "transformers-course/0.1 (education)"})
            data = urllib.request.urlopen(req, timeout=timeout).read()
            path.write_bytes(data)
            return data
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError


def books(n_books: int) -> list[dict]:
    out, page = [], 1
    while len(out) < n_books:
        listing = json.loads(fetch(LISTING.format(page), DIR / f"gutendex_{page}.json"))
        for b in listing["results"]:
            fmt = b["formats"]
            url = fmt.get("text/plain; charset=utf-8") or fmt.get("text/plain; charset=us-ascii") or fmt.get("text/plain")
            if b["copyright"] is False and url and not EXCLUDE.search(b["title"]):
                author = (b["authors"] or [{"name": "Various"}])[0]["name"]
                out.append({"id": b["id"], "title": b["title"], "author": author, "url": url})
        if not listing["next"]:
            break
        page += 1
    return out[:n_books]


def ingest(n_books: int) -> list[dict]:
    (DIR / "raw").mkdir(parents=True, exist_ok=True)
    docs = []
    for b in books(n_books):
        text = fetch(b["url"], DIR / "raw" / f"{b['id']}.txt").decode("utf-8", errors="replace")
        # "Last, First" -> "First Last". The anthology attribution line overrides it per poem.
        author = " ".join(reversed(b["author"].split(", ", 1))) if ", " in b["author"] else b["author"]
        for k, p in enumerate(split_anthology(gutenberg_body(text), author)):
            docs.append({"id": f"pg{b['id']}-{k}", "source": f"gutenberg:{b['id']}", "book": b["title"],
                         "book_author": author,
                         "licence": "public domain (Project Gutenberg)", **p})
    return docs


def words(t: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", t)


def lines(t: str) -> list[str]:
    return [l for l in t.split("\n") if l.strip()]


def is_prose(ls: list[str]) -> bool:
    long = sum(len(l.strip()) >= 60 for l in ls) > 0.5 * len(ls)
    return long and sum(bool(re.search(r"[,.;:!?\-—'\")]\s*$", l)) for l in ls) < 0.4 * len(ls)


# Each rule: (name, what it targets, keep predicate). Rules run in order; the funnel counts each one.
RULES = [
    ("annotations", "drop [Illustration], [Footnote] and transcriber lines; keep the poem",
     None),
    ("urls", "transcriber notes and site boilerplate: any URL. Verse in this corpus never contains one",
     lambda d: not re.search(r"https?://|www\.", d["text"])),
    ("min_lines", "fragments and stray headings: fewer than 4 lines",
     lambda d: len(lines(d["text"])) >= 4),
    ("prose", "prose (notes, biographies): over half the lines are 60+ characters and under 40 % end in punctuation."
     " Wrapped prose breaks mid-sentence. Long verse lines (Whitman) mostly end in punctuation",
     lambda d: not is_prose(lines(d["text"]))),
    ("max_lines", "over 120 lines: D1 packs one poem per T = 1024 window. This drops long poems on purpose",
     lambda d: len(lines(d["text"])) <= 120),
    ("min_words", "fewer than 20 words",
     lambda d: len(words(d["text"])) >= 20),
    ("max_symbols", "contents and index pages: over 20 % of non-space characters are not letters",
     lambda d: sum(not c.isalpha() for c in d["text"] if not c.isspace()) <= 0.2 * max(1, sum(not c.isspace() for c in d["text"]))),
    ("numbered_lines", "tables of contents: over 30 % of lines end in a number",
     lambda d: sum(bool(re.search(r"\d+\s*$", l)) for l in lines(d["text"])) <= 0.3 * len(lines(d["text"]))),
    ("english", "non-English text: under 12 % of words are common English function words",
     lambda d: sum(w.lower() in STOP for w in words(d["text"])) >= 0.12 * len(words(d["text"]))),
]
_ANNOT = re.compile(r"^\s*(\[(Illustration|Footnote|Transcriber)[^\n]*|_?Transcriber'?s? [Nn]ote.*)$", re.M)


def clean(docs: list[dict], show: str | None = None) -> tuple[list[dict], list[dict]]:
    funnel = [{"stage": "split into poems", "docs": len(docs), "words": sum(len(words(d["text"])) for d in docs)}]
    for d in docs:
        d["text"] = normalize_text(_ANNOT.sub("", d["text"])).strip("\n")
    for name, why, keep in RULES:
        if keep is not None:
            removed = [d for d in docs if not keep(d)]
            docs = [d for d in docs if keep(d)]
            if show == name:
                for d in random.Random(0).sample(removed, min(20, len(removed))):
                    print(f"--- removed by {name}: {d['id']} {d['title']!r}\n{d['text'][:300]}\n")
        funnel.append({"stage": name, "why": why, "docs": len(docs), "words": sum(len(words(d["text"])) for d in docs)})
    return docs, funnel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", type=int, default=40)
    ap.add_argument("--show", default=None, help="print 20 documents that this rule removed")
    args = ap.parse_args()
    docs = ingest(args.books)
    (DIR / "verse_raw.jsonl").write_text("".join(json.dumps(d) + "\n" for d in docs), encoding="utf-8")
    kept, funnel = clean(docs, args.show)
    (DIR / "verse_clean.jsonl").write_text("".join(json.dumps(d) + "\n" for d in kept), encoding="utf-8")
    (DIR / "funnel_clean.json").write_text(json.dumps(funnel, indent=1))
    print(f"{'stage':>18} {'docs':>7} {'words':>10}")
    for f in funnel:
        print(f"{f['stage']:>18} {f['docs']:>7} {f['words']:>10}")
    print(f"books: {len({d['source'] for d in kept})}")


if __name__ == "__main__":
    main()
