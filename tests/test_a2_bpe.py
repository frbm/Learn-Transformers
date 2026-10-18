import time
from collections import Counter
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from lm.tokenizer.bpe import Tokenizer, pretokenize, train_bpe

TEXT = Path("data/corpus.txt").read_text(encoding="utf-8")


def merge_word(w, pair, new_id):
    out, i = [], 0
    while i < len(w):
        if i + 1 < len(w) and (w[i], w[i + 1]) == pair:
            out.append(new_id); i += 2
        else:
            out.append(w[i]); i += 1
    return tuple(out)


def reference_bpe(text, n_merges):
    words = Counter(tuple(c.encode("utf-8")) for c in pretokenize(text))
    merges = []
    for k in range(n_merges):
        pairs = Counter()
        for w, c in words.items():
            for a, b in zip(w, w[1:]):
                pairs[a, b] += c
        if not pairs:
            break
        best = min(pairs, key=lambda p: (-pairs[p], p))
        merges.append(best)
        words = Counter({merge_word(w, best, 256 + k): c for w, c in words.items()})
    return merges


SMALL = TEXT[:50_000]
TOK = Tokenizer(train_bpe(TEXT[:300_000], vocab_size=2048))


def test_matches_reference():
    assert train_bpe(SMALL, vocab_size=256 + 300) == reference_bpe(SMALL, 300)


def test_matches_reference_on_runs():
    s = "aaaa aaa aa a " * 50 + "abab ababab " * 30
    assert train_bpe(s, vocab_size=256 + 40) == reference_bpe(s, 40)


def test_encoder_replays_training():
    merges = train_bpe(SMALL, vocab_size=256 + 300)
    words = Counter(tuple(c.encode("utf-8")) for c in pretokenize(SMALL))
    for k, pair in enumerate(merges):
        words = Counter({merge_word(w, pair, 256 + k): n for w, n in words.items()})
    assert len(Tokenizer(merges).encode(SMALL)) == sum(len(w) * n for w, n in words.items())


def test_more_merges_never_longer():
    merges = train_bpe(TEXT[:100_000], vocab_size=256 + 1000)
    held_out = TEXT[-20_000:]
    lengths = [len(Tokenizer(merges[:k]).encode(held_out)) for k in (0, 100, 300, 1000)]
    assert lengths == sorted(lengths, reverse=True)


EDGE = ["", " ", "\n\n\t ", "é", "é", "👩‍👩‍👧", "日本語のテキスト", "l'homme", "x" * 10_000, "\x00\r\n", "  both ends  "]


@pytest.mark.parametrize("s", EDGE)
def test_roundtrip_edge_cases(s):
    assert TOK.decode(TOK.encode(s)) == s.encode("utf-8")


@settings(max_examples=1000, deadline=None)
@given(st.text())
def test_roundtrip_random(s):
    assert TOK.decode(TOK.encode(s)) == s.encode("utf-8")


def test_byte_fallback():
    assert TOK.decode(list(range(256))) == bytes(range(256))


def test_save_load(tmp_path):
    TOK.save(tmp_path / "tok.json")
    assert Tokenizer.load(tmp_path / "tok.json").encode(SMALL) == TOK.encode(SMALL)


def test_special_tokens():
    tok = Tokenizer(TOK.merges, special=["<|endoftext|>", "[CHORUS]"])
    text = "verse one<|endoftext|>[CHORUS] la la [CHORUS]"
    ids = tok.encode(text)
    assert ids.count(tok.special["<|endoftext|>"]) == 1
    assert ids.count(tok.special["[CHORUS]"]) == 2
    assert tok.decode(ids) == text.encode("utf-8")
    assert min(tok.special.values()) == 256 + len(TOK.merges)


@pytest.mark.slow
def test_speed_16k():
    t0 = time.perf_counter()
    train_bpe(TEXT, vocab_size=16_384)
    assert time.perf_counter() - t0 < 300
