import numpy as np

V = 256


def window_keys(x: np.ndarray, n: int) -> np.ndarray:
    """Every length-n window of x as one int64 in base 256 (Horner, n vector passes)."""
    assert 1 <= n <= 7, "256**8 overflows int64: hash the windows instead"
    assert x.dtype == np.int64, "uint8 products overflow silently"
    m = len(x) - n + 1
    k = np.zeros(m, dtype=np.int64)
    for j in range(n):
        k = k * V + x[j : j + m]
    return k


def lookup(keys: np.ndarray, table_keys: np.ndarray, table_counts: np.ndarray) -> np.ndarray:
    """Counts of keys in a sorted table, 0 if absent. O(M log K)."""
    i = np.minimum(np.searchsorted(table_keys, keys), len(table_keys) - 1)
    return np.where(table_keys[i] == keys, table_counts[i], 0)


def sharded_unique(keys: np.ndarray, n_shards: int) -> tuple[np.ndarray, np.ndarray]:
    """np.unique by shard (key % n_shards). This is the map-reduce pattern for counts that do not fit in RAM."""
    parts = [np.unique(keys[keys % n_shards == s], return_counts=True) for s in range(n_shards)]
    k = np.concatenate([p[0] for p in parts])
    c = np.concatenate([p[1] for p in parts])
    order = np.argsort(k, kind="stable")
    return k[order], c[order]


class NGram:
    def __init__(self, n: int, alpha: float):
        self.n, self.alpha = n, alpha

    def fit(self, x: np.ndarray) -> "NGram":
        full = window_keys(x, self.n)
        self.full_k, self.full_c = np.unique(full, return_counts=True)
        # Count contexts from the same windows. Then sum_c N(h, c) = N(h) exactly.
        self.ctx_k, self.ctx_c = np.unique(full // V, return_counts=True)
        return self

    def nll(self, x: np.ndarray) -> float:
        """Total NLL (nats) of x[n-1:]. Each byte gets its n-1 predecessors as context."""
        full = window_keys(x, self.n)
        num = lookup(full, self.full_k, self.full_c) + self.alpha
        den = lookup(full // V, self.ctx_k, self.ctx_c) + self.alpha * V
        return float(-(np.log(num) - np.log(den)).sum())

    def next_logprobs(self, h) -> np.ndarray:
        """log q(. | h) over the 256 bytes. h is the last n-1 bytes."""
        hk = 0
        for b in h:
            hk = hk * V + int(b)
        num = lookup(hk * V + np.arange(V, dtype=np.int64), self.full_k, self.full_c) + self.alpha
        den = lookup(np.array([hk]), self.ctx_k, self.ctx_c)[0] + self.alpha * V
        return np.log(num) - np.log(den)


def sample(model: NGram, prompt: bytes, n_new: int, tau: float = 1.0, seed: int = 0) -> str:
    """The prompt must have at least n-1 bytes. Gumbel-max: argmax(z + G) ~ softmax(z), without normalisation."""
    rng = np.random.default_rng(seed)
    x = list(prompt)
    for _ in range(n_new):
        h = x[len(x) - (model.n - 1) :] if model.n > 1 else []
        z = model.next_logprobs(h) / tau
        x.append(int(np.argmax(z + rng.gumbel(size=V))))
    return bytes(x).decode("utf-8", errors="replace")
