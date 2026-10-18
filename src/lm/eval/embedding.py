"""F1 sample bank and embedding tools: sampling with log q, cached sentence embeddings, k-means quantisation."""
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from lm.verse.common import decode, header_text, load_tokenizer
from lm.infer.generate import generate

EMB_CACHE = Path("runs/f1/emb_cache")
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@torch.no_grad()
def sample_set(model, headers: list[dict], n_per_header: int, sampler, tok=None, max_new: int = 128,
               batch_size: int = 64, generator=None, instruction: str = "") -> list[dict]:
    """{"header", "poem", "logq", "n_tokens", "finished"}. logq is under the decoded q, eos term included."""
    tok = tok or load_tokenizer()
    eos = tok.special["<|endoftext|>"]
    jobs = [(h, tok.encode(header_text(h, h["lines"], tok, instruction))) for h in headers for _ in range(n_per_header)]
    out = []
    for b in range(0, len(jobs), batch_size):
        chunk = jobs[b : b + batch_size]
        seqs, logq = generate(model, [p for _, p in chunk], max_new, sampler, eos, generator=generator,
                              return_logq=True)
        for (h, _), s, lq in zip(chunk, seqs, logq):
            fin = len(s) < max_new
            out.append({"header": h, "poem": decode(tok, s), "logq": lq, "n_tokens": len(s) + fin, "finished": fin,
                        "ids": s})
    return out


def _hash_embed(texts: list[str], dim: int = 1024) -> Tensor:
    """Fallback embedding: hashed word unigrams and bigrams, TF weights. Deterministic, no download."""
    X = torch.zeros(len(texts), dim, dtype=torch.float64)
    for i, t in enumerate(texts):
        w = t.lower().split()
        for g in w + [a + " " + b for a, b in zip(w, w[1:])]:
            X[i, int(hashlib.md5(g.encode()).hexdigest()[:8], 16) % dim] += 1.0
    return F.normalize(X, dim=1)


def embed(texts: list[str], model_name: str = EMB_MODEL, batch_size: int = 128) -> Tensor:
    """Normalised sentence embeddings (n, k), cached on disk by a hash of the texts and the model name."""
    key = hashlib.sha1(("\x00".join(texts) + model_name).encode()).hexdigest()[:16]
    path = EMB_CACHE / f"{key}.pt"
    if path.exists():
        return torch.load(path)
    try:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
        E = torch.from_numpy(st.encode(texts, batch_size=batch_size, normalize_embeddings=True)).double()
    except ImportError:
        E = _hash_embed(texts)
    EMB_CACHE.mkdir(parents=True, exist_ok=True)
    torch.save(E, path)
    return E


def kmeans(X: Tensor, k: int, iters: int = 50, seed: int = 0) -> Tensor:
    """Lloyd's algorithm with k-means++ seeding. Returns labels (n,)."""
    g = torch.Generator().manual_seed(seed)
    X = X.double()
    C = X[torch.randint(len(X), (1,), generator=g)]
    for _ in range(1, k):
        d2 = torch.cdist(X, C).min(1).values ** 2
        C = torch.cat([C, X[torch.multinomial(d2 / d2.sum(), 1, generator=g)]])
    for _ in range(iters):
        lab = torch.cdist(X, C).argmin(1)
        counts = torch.bincount(lab, minlength=k).clamp(min=1)[:, None]
        new = torch.zeros_like(C).index_add_(0, lab, X) / counts
        empty = torch.bincount(lab, minlength=k) == 0
        new[empty] = C[empty]                                # keep an empty centre where it is
        if torch.allclose(new, C):
            break
        C = new
    return torch.cdist(X, C).argmin(1)


def quantise(emb_p: Tensor, emb_q: Tensor, k: int, seed: int = 0) -> tuple[Tensor, Tensor]:
    """k-means on the union, then the two cluster histograms (each sums to 1)."""
    lab = kmeans(torch.cat([emb_p, emb_q]), k, seed=seed)
    hp = torch.bincount(lab[: len(emb_p)], minlength=k).double()
    hq = torch.bincount(lab[len(emb_p):], minlength=k).double()
    return hp / hp.sum(), hq / hq.sum()


def divpo_pair(emb: Tensor, reward: Tensor, threshold: float) -> tuple[int, int] | None:
    """DivPO pair for one prompt. Chosen: the sample above the threshold farthest (mean cosine distance) from
    the other samples. Rejected: the sample below the threshold closest to the others. None if a side is empty."""
    dist = 1 - emb @ emb.T
    mean_d = dist.sum(1) / (len(emb) - 1)
    hi, lo = reward >= threshold, reward < threshold
    if not hi.any() or not lo.any():
        return None
    chosen = int(mean_d.masked_fill(~hi, float("-inf")).argmax())
    rejected = int(mean_d.masked_fill(~lo, float("inf")).argmin())
    return chosen, rejected
