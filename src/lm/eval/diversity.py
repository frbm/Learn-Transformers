"""F1 diversity metrics: surface n-gram scores, Vendi, divergence frontiers and MAUVE, exact MI, Gibbs tilt."""
import math
from collections import Counter

import torch
from torch import Tensor


def _ngrams(words: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def distinct_n(texts: list[str], n: int) -> float:
    grams = [g for t in texts for g in _ngrams(t.split(), n)]
    return len(set(grams)) / len(grams) if grams else 0.0


def bleu(hyp: str, refs: list[str], max_n: int = 4) -> float:
    """Corpus-free sentence BLEU. Clipped counts, geometric mean, brevity penalty to the closest reference length."""
    h, rs = hyp.split(), [r.split() for r in refs]
    if not h or not rs:
        return 0.0
    log_p = 0.0
    for n in range(1, max_n + 1):
        hc = Counter(_ngrams(h, n))
        total = sum(hc.values())
        if total == 0:
            return 0.0
        best = Counter()
        for r in rs:
            best |= Counter(_ngrams(r, n))                   # | keeps the max count per n-gram
        match = sum(min(c, best[g]) for g, c in hc.items())
        if match == 0:
            return 0.0
        log_p += math.log(match / total) / max_n
    r_len = min((len(r) for r in rs), key=lambda L: (abs(L - len(h)), L))
    bp = 1.0 if len(h) > r_len else math.exp(1 - r_len / len(h))
    return bp * math.exp(log_p)


def self_bleu(texts: list[str], max_n: int = 4) -> float:
    return sum(bleu(t, texts[:i] + texts[i + 1 :], max_n) for i, t in enumerate(texts)) / len(texts)


def vendi(K: Tensor) -> float:
    lam = torch.linalg.eigvalsh(K / K.shape[0]).clamp(min=0)
    lam = lam[lam > 0]
    return math.exp(-(lam * lam.log()).sum().item())


def kl(p: Tensor, q: Tensor) -> float:
    m = p > 0
    return (p[m] * (p[m].log() - q[m].log())).sum().item()


def divergence_curve(p: Tensor, q: Tensor, lambdas, c: float = 5.0) -> list[tuple[float, float]]:
    """(exp(-c KL(q || R)), exp(-c KL(p || R))) for R = lam p + (1 - lam) q (Proposition 3)."""
    out = []
    for lam in lambdas:
        R = lam * p + (1 - lam) * q
        out.append((math.exp(-c * kl(q, R)), math.exp(-c * kl(p, R))))
    return out


def mauve(p: Tensor, q: Tensor, c: float = 5.0, n_grid: int = 99) -> float:
    lams = torch.linspace(0, 1, n_grid + 2, dtype=torch.float64)[1:-1].tolist()
    pts = sorted([(0.0, 1.0), *divergence_curve(p, q, lams, c), (1.0, 0.0)])
    return sum((x1 - x0) * (y0 + y1) / 2 for (x0, y0), (x1, y1) in zip(pts, pts[1:]))


def mi_estimate(logq: Tensor, cond: Tensor, log_prior: Tensor) -> float:
    """Proposition 7. logq (n, C) = log q(y_i | c) for every c; cond (n,) = the c that generated y_i."""
    log_marg = torch.logsumexp(logq + log_prior, dim=1)
    return (logq.gather(1, cond[:, None])[:, 0] - log_marg).mean().item()


def mi_standard_error(logq: Tensor, cond: Tensor, log_prior: Tensor) -> float:
    terms = logq.gather(1, cond[:, None])[:, 0] - torch.logsumexp(logq + log_prior, dim=1)
    return (terms.std() / math.sqrt(len(terms))).item()


def tilt(log_ref: Tensor, r: Tensor, t: float) -> Tensor:
    """log pi_t with pi_t ∝ pi_ref exp(t r), normalised in log space."""
    return torch.log_softmax(log_ref + t * r, dim=0)


def entropy_estimate(logq: Tensor, n_tokens: Tensor) -> tuple[float, float]:
    """Proposition 1 per token, as a ratio of sums: (-sum log q) / sum n_tokens, and its delta-method s.e."""
    x, n = -logq.double(), n_tokens.double()
    H = (x.sum() / n.sum()).item()
    resid = (x - H * n) / n.mean()
    return H, (resid.std() / math.sqrt(len(x))).item() if len(x) > 1 else float("nan")


def pareto_front(points: list[tuple[float, float]]) -> list[int]:
    """Indices of points not dominated when both coordinates are maximised, sorted by the first coordinate."""
    idx = [i for i, (a, b) in enumerate(points)
           if not any((a2 >= a and b2 >= b) and (a2 > a or b2 > b) for a2, b2 in points)]
    return sorted(idx, key=lambda i: points[i][0])


def tv(p: Tensor, q: Tensor) -> float:
    return 0.5 * (p - q).abs().sum().item()
