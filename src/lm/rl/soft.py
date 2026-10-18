"""R1 §5. The token-tree MDP: Doob h-transform of the reference by exp(r / beta). Prefixes are tuples of token ids."""
import math

import torch
from torch import Tensor


def log_h_function(ref: dict[tuple, Tensor], reward: dict[tuple, float], beta: float) -> dict[tuple, float]:
    """log h by the soft Bellman equation, leaves first. Log space: exp(r / beta) overflows for small beta."""
    log_h = {y: r / beta for y, r in reward.items()}
    for p in sorted(ref, key=len, reverse=True):
        child = torch.tensor([log_h[p + (a,)] for a in range(len(ref[p]))], dtype=torch.float64)
        log_h[p] = torch.logsumexp(ref[p].double().log() + child, 0).item()
    return log_h


def h_function(ref: dict[tuple, Tensor], reward: dict[tuple, float], beta: float) -> dict[tuple, float]:
    return {p: math.exp(v) for p, v in log_h_function(ref, reward, beta).items()}


def tilted_policy(ref: dict[tuple, Tensor], h: dict[tuple, float]) -> dict[tuple, Tensor]:
    """pi*(a | s) = pi_ref(a | s) h(s.a) / h(s). It sums to 1 because h is a martingale."""
    return {p: q * torch.tensor([h[p + (a,)] for a in range(len(q))], dtype=q.dtype) / h[p] for p, q in ref.items()}
