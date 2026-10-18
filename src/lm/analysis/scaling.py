"""B4. Exact FLOP count, compute-optimal allocation, Huber fit of L(N, D) = E + A N^-alpha + B D^-beta, bootstrap."""
import itertools

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from lm.model import Config, expected_params

NAMES = ("E", "A", "B", "alpha", "beta")


def matmul_params(cfg: Config) -> tuple[int, int]:
    """(weights in the blocks' matmuls, weights in the readout). Norm gains and the embedding lookup cost no matmul."""
    d, dh, h = cfg.d_model, cfg.d_head, cfg.mlp_hidden
    attn = d * cfg.n_heads * dh + 2 * d * cfg.n_kv_heads * dh + cfg.n_heads * dh * d
    mlp = 3 * d * h if cfg.use_mlp else 0
    return cfg.n_layers * (attn + mlp), d * cfg.vocab_size


def param_counts(cfg: Config) -> dict[str, int]:
    """Both conventions. Total counts the tied embedding once. Non-embedding drops it (Kaplan)."""
    total = expected_params(cfg)
    return {"non_embedding": total - cfg.vocab_size * cfg.d_model, "total": total}


def flops_per_token(cfg: Config, n_ctx: int, training: bool = True) -> float:
    """Matmuls (2 FLOPs per weight) + attention (QK^T and AV: 4 n_ctx n_heads d_head) + readout.

    n_ctx is the mean number of keys a query attends: T for naive attention, about T / 2 for a causal kernel
    that skips masked blocks. Backward costs twice the forward for every term.
    """
    blocks, readout = matmul_params(cfg)
    fwd = 2 * (blocks + readout) + cfg.n_layers * 4 * n_ctx * cfg.n_heads * cfg.d_head
    return 3.0 * fwd if training else float(fwd)


def compute_optimal(C: float, A: float, B: float, alpha: float, beta: float) -> tuple[float, float]:
    """Proposition 2 under C = 6 N D."""
    G = (alpha * A / (beta * B)) ** (1 / (alpha + beta))
    N = G * (C / 6) ** (beta / (alpha + beta))
    return N, C / (6 * N)


def _objective(x: np.ndarray, lN: np.ndarray, lD: np.ndarray, lL: np.ndarray, delta: float):
    """Huber loss on log residuals and its gradient. x = (a, b, e, alpha, beta) with A = e^a, B = e^b, E = e^e."""
    a, b, e, al, be = x
    z = np.stack([a - al * lN, b - be * lD, np.full_like(lN, e)])
    r = logsumexp(z, axis=0) - lL
    hub = np.where(np.abs(r) <= delta, 0.5 * r**2, delta * (np.abs(r) - 0.5 * delta))
    w = softmax(z, axis=0) * np.clip(r, -delta, delta)
    grad = np.array([w[0].sum(), w[1].sum(), w[2].sum(), -(w[0] * lN).sum(), -(w[1] * lD).sum()])
    # Divide by delta: the objective is then O(n) and L-BFGS tolerances behave.
    return hub.sum() / delta, grad / delta


def _grid() -> np.ndarray:
    return np.array(list(itertools.product([0, 5, 10, 15, 20, 25], [0, 5, 10, 15, 20, 25], [-1, -0.5, 0, 0.5, 1],
                                           [0, 0.5, 1, 1.5, 2], [0, 0.5, 1, 1.5, 2])), dtype=float)


def _grid_losses(X: np.ndarray, lN, lD, lL, delta) -> np.ndarray:
    """Huber objective of every grid point at once: (n_grid, n_runs) arrays, no Python loop."""
    z = np.stack([X[:, :1] - X[:, 3:4] * lN, X[:, 1:2] - X[:, 4:5] * lD, np.broadcast_to(X[:, 2:3], (len(X), lN.size))])
    r = np.abs(logsumexp(z, axis=0) - lL)
    return np.where(r <= delta, 0.5 * r**2, delta * (r - 0.5 * delta)).sum(1)


def _fit(lN, lD, lL, delta, starts: np.ndarray) -> np.ndarray:
    bounds = [(None, None)] * 3 + [(0, 5), (0, 5)]
    best = None
    for x0 in starts:
        res = minimize(_objective, x0, args=(lN, lD, lL, delta), jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-12})
        if best is None or res.fun < best.fun:
            best = res
    return best.x


def _to_dict(x: np.ndarray) -> dict:
    a, b, e, al, be = x
    return {"E": float(np.exp(e)), "A": float(np.exp(a)), "B": float(np.exp(b)), "alpha": float(al), "beta": float(be)}


def fit_scaling_law(N, D, L, delta: float = 1e-3, n_starts: int = 20, extra_starts: np.ndarray | None = None) -> dict:
    """Chinchilla approach 3. Screen the whole grid in one vectorised pass, then polish the best starts with L-BFGS-B."""
    lN, lD, lL = (np.log(np.asarray(v, dtype=float)) for v in (N, D, L))
    X = _grid()
    starts = X[np.argsort(_grid_losses(X, lN, lD, lL, delta))[:n_starts]]
    if extra_starts is not None:
        starts = np.vstack([extra_starts, starts])
    return _to_dict(_fit(lN, lD, lL, delta, starts))


def _to_x(p: dict) -> np.ndarray:
    # A fit at the boundary E -> 0 underflows exp(e) to 0.0. log(0) = -inf would poison every warm start,
    # and the bootstrap would return one point: an interval of width 0. Clamp to the smallest normal float.
    tiny = np.finfo(float).tiny
    return np.array([np.log(max(p["A"], tiny)), np.log(max(p["B"], tiny)), np.log(max(p["E"], tiny)),
                     p["alpha"], p["beta"]])


def bootstrap_fits(N, D, L, n_boot: int = 1000, seed: int = 0, delta: float = 1e-3) -> list[dict]:
    """Refits on runs resampled with replacement."""
    N, D, L = (np.asarray(v, dtype=float) for v in (N, D, L))
    rng = np.random.default_rng(seed)
    x_full = _to_x(fit_scaling_law(N, D, L, delta))
    fits = []
    for _ in range(n_boot):
        i = rng.integers(0, len(N), len(N))
        # Warm start at the full fit, plus two grid starts. One start can sit in the wrong basin.
        fits.append(fit_scaling_law(N[i], D[i], L[i], delta, n_starts=2, extra_starts=x_full[None]))
    return fits


def bootstrap(N, D, L, n_boot: int = 1000, level: float = 0.95, seed: int = 0, delta: float = 1e-3) -> dict:
    """Percentile intervals per parameter. Also a = beta / (alpha + beta), the exponent of N*(C)."""
    fits = bootstrap_fits(N, D, L, n_boot, seed, delta)
    arr = np.array([[p[k] for k in NAMES] + [p["beta"] / (p["alpha"] + p["beta"])] for p in fits])
    q = np.quantile(arr, [(1 - level) / 2, (1 + level) / 2], axis=0)
    return {k: (float(q[0, j]), float(q[1, j])) for j, k in enumerate(NAMES + ("a",))}
