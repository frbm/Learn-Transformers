"""D3 Task 1. A programmatic verse reward from the CMU Pronouncing Dictionary (pip install pronouncing)."""
import re
from functools import lru_cache

import pronouncing

DEFAULT_WEIGHTS = {"lines": 0.25, "rhyme": 0.35, "metre": 0.2, "repetition": 0.2}
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)*")


def normalise(word: str) -> str:
    m = _WORD.findall(word.lower().replace("’", "'"))
    return m[-1] if m else ""


@lru_cache(maxsize=None)
def _phones(word: str) -> str | None:
    p = pronouncing.phones_for_word(normalise(word))
    return p[0] if p else None


@lru_cache(maxsize=None)
def syllable_count(word: str) -> int | None:
    p = _phones(word)
    return pronouncing.syllable_count(p) if p else None


def _vowel_groups(word: str) -> int:
    # Fallback for out-of-vocabulary words (archaic spellings). Drop a silent final "e".
    w = normalise(word)
    n = len(re.findall(r"[aeiouy]+", w))
    return max(1, n - (w.endswith("e") and not w.endswith("le") and n > 1))


@lru_cache(maxsize=None)
def rhyme_part(word: str) -> str | None:
    p = _phones(word)
    # Drop the stress digits. Then "AE1 T" and "AE2 T" compare equal.
    return re.sub(r"\d", "", pronouncing.rhyming_part(p)) if p else None


def rhymes(a: str, b: str) -> bool:
    ra, rb = rhyme_part(a), rhyme_part(b)
    return ra is not None and ra == rb and normalise(a) != normalise(b)


def poem_lines(poem: str) -> list[str]:
    return [l.strip() for l in poem.splitlines() if _WORD.search(l.lower())]


def _line_syllables(line: str) -> int:
    return sum(syllable_count(w) or _vowel_groups(w) for w in line.split() if normalise(w))


def _final_words(lines: list[str]) -> list[str]:
    return [normalise(l.split()[-1]) or normalise(l) for l in lines]


def verse_reward(n_lines: int, poem: str, weights: dict | None = None) -> dict[str, float]:
    w = weights or DEFAULT_WEIGHTS
    lines = poem_lines(poem)
    L = len(lines)
    out = dict.fromkeys(("lines", "rhyme", "metre", "repetition"), 0.0)
    if L:
        d = abs(L - n_lines)
        out["lines"] = 1.0 if d <= 1 else max(0.0, 1 - (d - 1) / max(n_lines, 1))
    if L >= 2:
        # Score repetition only from two lines on. Else a one-line "poem" gets it for free.
        ends = _final_words(lines)
        out["repetition"] = len(set(ends)) / L
        hit = [rhymes(ends[i], ends[i + 1]) or (i + 2 < L and rhymes(ends[i], ends[i + 2])) for i in range(L - 1)]
        out["rhyme"] = sum(hit) / (L - 1)
        syl = [_line_syllables(l) for l in lines]
        mean = sum(syl) / L
        std = (sum((s - mean) ** 2 for s in syl) / L) ** 0.5
        out["metre"] = max(0.0, 1 - std / max(mean, 1))
    out["total"] = sum(w[k] * out[k] for k in w) / sum(w.values())
    return out


def naive_rhyme_reward(poem: str) -> float:
    """Task 5, hackable on purpose: adjacent final words with the same rhyme part. Identical words count."""
    ends = _final_words(poem_lines(poem))
    if len(ends) < 2:
        return 0.0
    rp = [rhyme_part(e) for e in ends]
    return sum(a is not None and a == b for a, b in zip(rp, rp[1:])) / (len(ends) - 1)
