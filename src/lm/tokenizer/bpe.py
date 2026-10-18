import heapq
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import regex

PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_PRETOK = regex.compile(PATTERN)


def pretokenize(text: str) -> list[str]:
    return _PRETOK.findall(text)


def _pairs(w) -> Counter:
    return Counter(zip(w, w[1:]))


def _merge(w: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    a, b = pair
    out, i, n = [], 0, len(w)
    while i < n:
        if i + 1 < n and w[i] == a and w[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(w[i])
            i += 1
    return out


def train_bpe(text: str, vocab_size: int) -> list[tuple[int, int]]:
    """Incremental BPE. Each merge touches only the chunks that contain the pair."""
    chunk_freq = Counter(pretokenize(text))
    words = [list(c.encode("utf-8")) for c in chunk_freq]
    freqs = list(chunk_freq.values())

    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    where: dict[tuple[int, int], set[int]] = defaultdict(set)
    for wid, (w, f) in enumerate(zip(words, freqs)):
        for p, c in _pairs(w).items():
            pair_counts[p] += c * f
            where[p].add(wid)

    # Lazy-deletion max-heap. (-count, pair) order also gives the tie rule: smallest pair first.
    heap = [(-c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)

    merges = []
    for k in range(vocab_size - 256):
        while heap:
            neg, best = heapq.heappop(heap)
            if pair_counts.get(best, 0) == -neg and neg < 0:
                break
        else:
            break
        new_id = 256 + k
        merges.append(best)

        touched = set()
        for wid in list(where[best]):
            w, f = words[wid], freqs[wid]
            old = _pairs(w)
            nw = _merge(w, best, new_id)
            new = _pairs(nw)
            words[wid] = nw
            for p in old.keys() | new.keys():
                d = new.get(p, 0) - old.get(p, 0)
                if d:
                    pair_counts[p] += d * f
                    touched.add(p)
                if p not in new:
                    where[p].discard(wid)
                elif p not in old:
                    where[p].add(wid)
        del where[best]
        pair_counts.pop(best, None)
        for p in touched:
            c = pair_counts.get(p, 0)
            if c > 0:
                heapq.heappush(heap, (-c, p))
            else:
                pair_counts.pop(p, None)
    return merges


class Tokenizer:
    def __init__(self, merges: list[tuple[int, int]], special: list[str] = ()):
        self.merges = [tuple(m) for m in merges]
        self.ranks = {p: k for k, p in enumerate(self.merges)}
        self.vocab = [bytes([b]) for b in range(256)]
        for a, b in self.merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        base = 256 + len(self.merges)
        self.special = {s: base + i for i, s in enumerate(special)}
        self.vocab += [s.encode("utf-8") for s in special]
        self._split = (
            regex.compile("(" + "|".join(regex.escape(s) for s in sorted(special, key=len, reverse=True)) + ")")
            if special
            else None
        )
        # The cache is per instance, so that two tokenisers never share entries.
        self._encode_chunk = lru_cache(maxsize=1 << 18)(self._encode_chunk_uncached)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _encode_chunk_uncached(self, chunk: str) -> tuple[int, ...]:
        w = list(chunk.encode("utf-8"))
        ranks, inf = self.ranks, float("inf")
        while len(w) > 1:
            pair = min(zip(w, w[1:]), key=lambda p: ranks.get(p, inf))
            r = ranks.get(pair)
            if r is None:
                break
            w = _merge(w, pair, 256 + r)
        return tuple(w)

    def _encode_ordinary(self, text: str, out: list[int]) -> None:
        enc = self._encode_chunk
        for c in _PRETOK.findall(text):
            out.extend(enc(c))

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        if self._split is None:
            self._encode_ordinary(text, out)
            return out
        for i, piece in enumerate(self._split.split(text)):
            # A split with one capturing group puts the matches at the odd indices.
            if i % 2:
                out.append(self.special[piece])
            elif piece:
                self._encode_ordinary(piece, out)
        return out

    def decode(self, ids: list[int]) -> bytes:
        v = self.vocab
        return b"".join(v[i] for i in ids)

    def save(self, path) -> None:
        Path(path).write_text(json.dumps({"merges": self.merges, "special": list(self.special)}))

    @classmethod
    def load(cls, path) -> "Tokenizer":
        d = json.loads(Path(path).read_text())
        return cls([tuple(m) for m in d["merges"]], d["special"])
