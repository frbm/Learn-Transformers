"""A2 Task 6: the fibre d^{-1}(s) under a unigram model. Forward, Viterbi, backward, one EM step."""
import math
from collections import Counter, defaultdict

import numpy as np

NEG_INF = -math.inf


def _lae(a: float, b: float) -> float:
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def surface_logphi(vocab: list[bytes], counts: Counter, alpha: float = 0.5) -> dict[bytes, float]:
    """Add-alpha unigram over the vocabulary. Merge duplicate surface strings with logaddexp."""
    total = sum(counts.values()) + alpha * len(vocab)
    out: dict[bytes, float] = {}
    for i, s in enumerate(vocab):
        lp = math.log((counts.get(i, 0) + alpha) / total)
        out[s] = _lae(out.get(s, NEG_INF), lp)
    return out


def _edges_into(chunk: bytes, j: int, logphi, max_len: int):
    for i in range(max(0, j - max_len), j):
        lp = logphi.get(chunk[i:j])
        if lp is not None:
            yield i, lp


def forward(chunk: bytes, logphi: dict[bytes, float], max_len: int) -> list[float]:
    a = [NEG_INF] * (len(chunk) + 1)
    a[0] = 0.0
    for j in range(1, len(chunk) + 1):
        acc = NEG_INF
        for i, lp in _edges_into(chunk, j, logphi, max_len):
            acc = _lae(acc, a[i] + lp)
        a[j] = acc
    return a


def backward(chunk: bytes, logphi: dict[bytes, float], max_len: int) -> list[float]:
    n = len(chunk)
    b = [NEG_INF] * (n + 1)
    b[n] = 0.0
    for i in range(n - 1, -1, -1):
        acc = NEG_INF
        for j in range(i + 1, min(n, i + max_len) + 1):
            lp = logphi.get(chunk[i:j])
            if lp is not None:
                acc = _lae(acc, lp + b[j])
        b[i] = acc
    return b


def log_total(chunk: bytes, logphi: dict[bytes, float], max_len: int) -> float:
    return forward(chunk, logphi, max_len)[-1]


def viterbi(chunk: bytes, logphi: dict[bytes, float], max_len: int) -> list[bytes]:
    n = len(chunk)
    best = [NEG_INF] * (n + 1)
    back = [0] * (n + 1)
    best[0] = 0.0
    for j in range(1, n + 1):
        for i, lp in _edges_into(chunk, j, logphi, max_len):
            if best[i] + lp > best[j]:
                best[j], back[j] = best[i] + lp, i
    out, j = [], n
    while j > 0:
        out.append(chunk[back[j] : j])
        j = back[j]
    return out[::-1]


def em_step(chunks: Counter, logphi: dict[bytes, float], max_len: int) -> tuple[dict[bytes, float], float]:
    """One EM step on chunk frequencies. Returns the new log-phi and the log-likelihood under the old one."""
    expected: dict[bytes, float] = defaultdict(float)
    loglik = 0.0
    for chunk, f in chunks.items():
        a, b = forward(chunk, logphi, max_len), backward(chunk, logphi, max_len)
        z = a[-1]
        loglik += f * z
        for j in range(1, len(chunk) + 1):
            for i, lp in _edges_into(chunk, j, logphi, max_len):
                expected[chunk[i:j]] += f * math.exp(a[i] + lp + b[j] - z)
    total = sum(expected.values())
    floor = math.log(1e-12)
    new = {s: (math.log(expected[s] / total) if expected.get(s, 0) > 0 else floor) for s in logphi}
    return new, loglik


def bpb(total_logprob_nats: float, n_bytes: int) -> float:
    return -total_logprob_nats / (np.log(2) * n_bytes)
