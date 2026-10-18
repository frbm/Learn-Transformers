"""R1 §7. Bernoulli bandits. Each algorithm runs many seeds in parallel: one row per seed."""
import numpy as np
import torch
from torch import Tensor


def run_bandits(algo: str, means: Tensor, T: int, seeds: int, seed: int = 0, eps: float = 0.1) -> Tensor:
    """Cumulative pseudo-regret sum_t (mu* - mu_{a_t}) -> (seeds, T)."""
    rng = np.random.default_rng(seed)
    mu = means.double().numpy()
    K, rows = len(mu), np.arange(seeds)
    n = np.zeros((seeds, K))
    s = np.zeros((seeds, K))
    gap = np.empty((seeds, T))
    for t in range(T):
        if algo == "eps_greedy":
            # Unpulled arms get +inf, so each arm is tried once before the greedy choice.
            est = np.where(n > 0, s / np.maximum(n, 1), np.inf)
            a = np.where(rng.random(seeds) < eps, rng.integers(0, K, seeds), est.argmax(1))
        elif algo == "ucb":
            bonus = np.sqrt(2 * np.log(t + 1) / np.maximum(n, 1))
            a = np.where(n > 0, s / np.maximum(n, 1) + bonus, np.inf).argmax(1)
        elif algo == "thompson":
            a = rng.beta(1 + s, 1 + n - s).argmax(1)
        else:
            raise ValueError(algo)
        x = rng.random(seeds) < mu[a]
        n[rows, a] += 1
        s[rows, a] += x
        gap[:, t] = mu.max() - mu[a]
    return torch.from_numpy(gap.cumsum(1))


def run_bandit(algo: str, means: Tensor, T: int, seed: int, eps: float = 0.1) -> Tensor:
    return run_bandits(algo, means, T, seeds=1, seed=seed, eps=eps)[0]


def ucb_bound(means: Tensor, T: int) -> Tensor:
    """Auer et al. 2002: sum_{gap > 0} 8 log t / gap + (1 + pi^2 / 3) sum gap, for t = 1..T."""
    gap = means.max() - means.double()
    gap = gap[gap > 0]
    t = torch.arange(1, T + 1, dtype=torch.float64)
    return 8 * torch.log(t) * (1 / gap).sum() + (1 + torch.pi ** 2 / 3) * gap.sum()
