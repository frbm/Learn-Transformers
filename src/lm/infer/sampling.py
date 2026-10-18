"""D4 sampler library. A processor maps logits (..., V) to logits with -inf outside the kept set."""
import math
from typing import Callable

import torch
from torch import Tensor

NEG_INF = float("-inf")


def temperature(logits: Tensor, T: float) -> Tensor:
    """softmax(z / T). T = 0 keeps the argmax only (greedy)."""
    if T == 0:
        return logits.masked_fill(logits < logits.amax(-1, keepdim=True), NEG_INF)
    return logits / T


def top_k(logits: Tensor, k: int) -> Tensor:
    kth = logits.topk(min(k, logits.shape[-1]), dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth, NEG_INF)


def _keep_sorted_prefix(logits: Tensor, order_key: Tensor, mass: float) -> Tensor:
    """Sort by order_key ascending. Keep the shortest prefix with probability mass >= mass."""
    probs = logits.softmax(-1, dtype=torch.float64 if logits.dtype == torch.float64 else torch.float32)
    idx = order_key.argsort(dim=-1)
    ps = probs.gather(-1, idx)
    # Keep position i iff the mass BEFORE it is < mass. This keeps the crossing token and never keeps nothing.
    keep_sorted = (ps.cumsum(-1) - ps) < mass
    keep = torch.zeros_like(keep_sorted).scatter(-1, idx, keep_sorted)
    return logits.masked_fill(~keep, NEG_INF)


def top_p(logits: Tensor, p: float) -> Tensor:
    return _keep_sorted_prefix(logits, -logits, p)


def min_p(logits: Tensor, p_min: float) -> Tensor:
    # p(a) >= p_min * max p  <=>  z_a >= z_max + log p_min. No softmax needed.
    return logits.masked_fill(logits < logits.amax(-1, keepdim=True) + math.log(p_min), NEG_INF)


def _entropy(logp: Tensor) -> Tensor:
    return -(logp.exp() * logp.nan_to_num(neginf=0.0)).sum(-1, keepdim=True)


def eta_cutoff(logits: Tensor, eta: float) -> Tensor:
    logp = logits.log_softmax(-1)
    thr = torch.minimum(torch.full_like(logp[..., :1], eta), math.sqrt(eta) * torch.exp(-_entropy(logp)))
    keep = (logp.exp() > thr) | (logits == logits.amax(-1, keepdim=True))
    return logits.masked_fill(~keep, NEG_INF)


def typical(logits: Tensor, tau: float) -> Tensor:
    logp = logits.log_softmax(-1)
    return _keep_sorted_prefix(logits, (-logp - _entropy(logp)).abs(), tau)


def truncation_kl(logits: Tensor, truncated: Tensor) -> Tensor:
    """KL(q || p) = -log p(S) for q = p restricted to S and renormalised (Proposition 3)."""
    logp = logits.log_softmax(-1).masked_fill(~torch.isfinite(truncated), NEG_INF)
    return -torch.logsumexp(logp, dim=-1)


class Sampler:
    """Apply the processors in order, then draw by Gumbel-max. Sampler([]) is ancestral sampling at T = 1."""

    def __init__(self, processors: list[Callable[[Tensor], Tensor]]):
        self.processors = list(processors)

    @classmethod
    def greedy(cls) -> "Sampler":
        return cls([lambda x: temperature(x, 0.0)])

    def process(self, logits: Tensor) -> Tensor:
        for f in self.processors:
            logits = f(logits)
        return logits

    def __call__(self, logits: Tensor, generator=None) -> Tensor:
        return self.draw(self.process(logits.float() if logits.dtype != torch.float64 else logits), generator)

    @staticmethod
    def draw(processed: Tensor, generator=None) -> Tensor:
        # argmax(z - log E) with E ~ Exp(1) is a sample from softmax(z). It needs no normalisation.
        e = torch.empty_like(processed).exponential_(generator=generator)
        return (processed - e.log()).argmax(-1)


def make_sampler(spec: str) -> Sampler:
    """Parse 'greedy', 'temp=0.8', 'top_p=0.9', 'temp=0.8,min_p=0.1'. The order of the terms is the order of use."""
    if spec == "greedy":
        return Sampler.greedy()
    procs = {"temp": temperature, "top_k": top_k, "top_p": top_p, "min_p": min_p, "eta": eta_cutoff, "typical": typical}
    out = []
    for term in filter(None, spec.split(",")):
        name, val = term.split("=")
        v = int(val) if name == "top_k" else float(val)
        out.append(lambda x, f=procs[name], v=v: f(x, v))
    return Sampler(out)
