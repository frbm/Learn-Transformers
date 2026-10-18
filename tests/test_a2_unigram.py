import math
from collections import Counter

from lm.tokenizer.unigram import em_step, log_total, viterbi


def segmentations(s: bytes, vocab):
    if not s:
        yield []
        return
    for j in range(1, len(s) + 1):
        if s[:j] in vocab:
            for rest in segmentations(s[j:], vocab):
                yield [s[:j]] + rest


LOGPHI = {bytes([b]): math.log(0.01) for b in b"abcd"} | {b"ab": math.log(0.2), b"bc": math.log(0.3), b"cd": math.log(0.1), b"abc": math.log(0.05)}


def test_forward_matches_enumeration():
    for s in (b"abcd", b"abcabc", b"dcba", b"aabbccdd"):
        brute = math.log(sum(math.exp(sum(LOGPHI[t] for t in seg)) for seg in segmentations(s, LOGPHI)))
        assert abs(log_total(s, LOGPHI, 3) - brute) < 1e-9


def test_viterbi_is_argmax():
    for s in (b"abcd", b"abcabc"):
        best = max(segmentations(s, LOGPHI), key=lambda seg: sum(LOGPHI[t] for t in seg))
        assert viterbi(s, LOGPHI, 3) == best


def test_em_does_not_decrease_likelihood():
    chunks = Counter({b"abcd": 5, b"abc": 3, b"cdab": 2})
    phi, ll = dict(LOGPHI), -math.inf
    for _ in range(5):
        phi, new_ll = em_step(chunks, phi, 3)
        assert new_ll >= ll - 1e-9
        ll = new_ll
