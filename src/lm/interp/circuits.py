"""E1 circuits: weights of heads, direct attribution, lenses, QK and OV circuits, patching."""
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from lm.interp.hooks import hooks, run_with_cache
from lm.model import GPT
from lm.model.attention import causal_mask
from lm.model.layers import rms_scale, work_dtype
from lm.model.rope import apply_rope

Metric = Callable[[Tensor], Tensor]


def head_weights(model: GPT, layer: int, head: int) -> dict[str, Tensor]:
    """W_Q, W_K, W_V (d, d_h), W_O (d_h, d), with the attn_norm gain folded in. g_q, g_k: the QK-norm gains.

    QK-norm divides q and k by their own RMS. That gain cannot fold into W_Q, so it stays separate.
    """
    at, g = model.blocks[layer].attn, model.blocks[layer].attn_norm.weight
    dh = model.cfg.d_head
    kv = head // (at.n_q // at.n_kv)
    rows = lambda w, h: w.weight[h * dh : (h + 1) * dh].T * g[:, None]
    return {k: v.detach() for k, v in dict(
        W_Q=rows(at.wq, head), W_K=rows(at.wk, kv), W_V=rows(at.wv, kv),
        W_O=at.wo.weight[:, head * dh : (head + 1) * dh].T, g_q=at.q_norm.weight, g_k=at.k_norm.weight).items()}


def head_outputs(model: GPT, cache: dict[str, Tensor], layer: int) -> Tensor:
    z = cache[f"blocks.{layer}.attn.hook_z"]
    W_O = model.blocks[layer].attn.wo.weight.view(model.cfg.d_model, model.cfg.n_heads, model.cfg.d_head)
    return torch.einsum("bthk,dhk->bthd", z, W_O.to(z.dtype))


def residual_components(model: GPT, cache: dict[str, Tensor]) -> dict[str, Tensor]:
    out = {"embed": cache["hook_embed"]}
    for l in range(model.cfg.n_layers):
        out[f"L{l}.attn"] = cache[f"blocks.{l}.hook_attn_out"]
        if model.cfg.use_mlp:
            out[f"L{l}.mlp"] = cache[f"blocks.{l}.hook_mlp_out"]
    return out


def unembed_direction(model: GPT, direction: Tensor) -> Tensor:
    """g ⊙ (W_U u): the residual direction that the logit direction u reads, before division by rho. (d,)"""
    return model.norm.weight * (direction.to(model.lm_head.weight.dtype) @ model.lm_head.weight)


def dla(model: GPT, cache: dict[str, Tensor], components: dict[str, Tensor], direction: Tensor) -> dict[str, Tensor]:
    """Direct logit attribution with the final rho frozen. Exact: the values sum to logits @ direction."""
    rho, w = cache["norm.hook_scale"], unembed_direction(model, direction).detach()
    return {k: (c.to(rho.dtype) / rho) @ w.to(rho.dtype) for k, c in components.items()}


def residual_stack(model: GPT, cache: dict[str, Tensor]) -> Tensor:
    """(L + 1, B, T, d): the stream before each block, then after the last."""
    L = model.cfg.n_layers
    return torch.stack([cache[f"blocks.{l}.hook_resid_pre"] for l in range(L)] + [cache[f"blocks.{L - 1}.hook_resid_post"]])


@torch.no_grad()
def logit_lens(model: GPT, cache: dict[str, Tensor]) -> Tensor:
    """(L + 1, B, T, V) log-probabilities. Each layer uses its own rho, as if it were the final stream."""
    lg = model.logits(residual_stack(model, cache))
    return F.log_softmax(lg.to(work_dtype(lg.dtype)), dim=-1)


def qk_pattern(model: GPT, cache: dict[str, Tensor], layer: int, head: int) -> Tensor:
    """(B, T, T). S_ij = n(y_i W_Q) R_{j-i} n(y_j W_K)^T * scale, with n the QK-norm and y the normed stream."""
    x = cache[f"blocks.{layer}.hook_resid_pre"]
    W, eps = head_weights(model, layer, head), model.blocks[layer].attn.q_norm.eps
    dt = work_dtype(x.dtype)
    y = x.to(dt) / rms_scale(x, model.cfg.norm_eps)
    q, k = y @ W["W_Q"].to(dt), y @ W["W_K"].to(dt)
    q = q / rms_scale(q, eps) * W["g_q"].to(dt)
    k = k / rms_scale(k, eps) * W["g_k"].to(dt)
    if model.rope is not None:
        q, k = apply_rope(q, *model.rope), apply_rope(k, *model.rope)
    T = x.shape[1]
    s = (q @ k.transpose(-1, -2)) * model.blocks[layer].attn.scale
    return s.masked_fill(~causal_mask(T, T, 0, x.device), float("-inf")).softmax(-1)


def ov_matrix(model: GPT, layer: int, head: int) -> Tensor:
    W = head_weights(model, layer, head)
    return W["W_V"] @ W["W_O"]


def qk_matrix(model: GPT, layer: int, head: int) -> Tensor:
    """W_Q diag(g_q g_k) W_K^T at offset 0. It ignores the QK-norm division, so use it for composition scores only."""
    W = head_weights(model, layer, head)
    return (W["W_Q"] * (W["g_q"] * W["g_k"])) @ W["W_K"].T


def copying_score(W_E: Tensor, W_OV: Tensor, W_U: Tensor) -> float:
    """sum(lambda) / sum(|lambda|) of W_E W_OV W_U. The d x d matrix W_OV W_U W_E has the same nonzero spectrum."""
    lam = torch.linalg.eigvals(W_OV @ W_U @ W_E)
    return (lam.real.sum() / lam.abs().sum()).item()


def model_copying_score(model: GPT, layer: int, head: int) -> float:
    W_U = model.lm_head.weight.T * model.norm.weight[:, None]
    return copying_score(model.tok_emb.weight.detach(), ov_matrix(model, layer, head), W_U.detach())


def composition_score(W_A: Tensor, W_B: Tensor) -> float:
    return ((W_A @ W_B).norm() / (W_A.norm() * W_B.norm())).item()


def composition_scores(model: GPT, l1: int, h1: int, l2: int, h2: int) -> dict[str, float]:
    """How much of head 1's output lands where head 2 reads: its queries, keys or values."""
    ov1, qk2, ov2 = ov_matrix(model, l1, h1), qk_matrix(model, l2, h2), ov_matrix(model, l2, h2)
    return {"Q": composition_score(ov1, qk2), "K": composition_score(ov1, qk2.T), "V": composition_score(ov1, ov2)}


def composition_null(d: int, d_h: int, n: int = 200, seed: int = 0) -> Tensor:
    """K-composition scores of random Gaussian factors with the true shapes. (n,)"""
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(n, *s, generator=g, dtype=torch.float64)
    ov, qk = r(d, d_h) @ r(d_h, d), r(d, d_h) @ r(d_h, d)
    prod = ov @ qk.transpose(-1, -2)
    return prod.norm(dim=(1, 2)) / (ov.norm(dim=(1, 2)) * qk.norm(dim=(1, 2)))


def prefix_matching_score(pattern: Tensor, half: int) -> float:
    """Mean attention from position i of the second copy to i - half + 1 (the token after the first occurrence)."""
    i = torch.arange(half, pattern.shape[-1], device=pattern.device)
    return pattern[..., i, i - half + 1].mean().item()


def previous_token_score(pattern: Tensor) -> float:
    i = torch.arange(1, pattern.shape[-1], device=pattern.device)
    return pattern[..., i, i - 1].mean().item()


def _replace(value: Tensor, index):
    def fn(o: Tensor) -> Tensor:
        if index is None:
            return value.to(o.dtype).expand_as(o)
        o = o.clone()
        o[index] = value[index].to(o.dtype)
        return o
    return fn


@torch.no_grad()
def patch(model: GPT, idx: Tensor, name: str, value: Tensor, metric: Metric, index=None) -> float:
    """Exact m(idx; do(a := value)). With index, only a[index] := value[index]."""
    with hooks(model, {name: _replace(value, index)}):
        return float(metric(model(idx)))


def _grad_run(model: GPT, idx: Tensor, names: list[str], metric: Metric) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
    """One forward and one backward. Returns m, the activations and dm/da (total derivatives)."""
    acts: dict[str, Tensor] = {}

    def keep(n):
        def fn(o):
            # Do not detach a node with a grad path: a detach here cuts the gradient to every earlier node.
            if not o.requires_grad:
                o = o.detach().requires_grad_(True)
            acts[n] = o
            return o
        return fn

    with torch.enable_grad(), hooks(model, {n: keep(n) for n in names}):
        m = metric(model(idx))
        grads = torch.autograd.grad(m, [acts[n] for n in names], allow_unused=True)
    grads = [torch.zeros_like(acts[n]) if g is None else g for n, g in zip(names, grads)]
    return m.detach(), {n: a.detach() for n, a in acts.items()}, dict(zip(names, grads))


def linear_patch(model: GPT, idx: Tensor, name: str, value: Tensor, metric: Metric) -> float:
    """First-order estimate m(idx) + <dm/da, value - a> (Proposition 6)."""
    m, acts, grads = _grad_run(model, idx, [name], metric)
    return m.item() + (grads[name] * (value.to(acts[name].dtype) - acts[name])).sum().item()


def _grid_dims(a: Tensor, name: str) -> tuple[int, ...]:
    """Axes summed out of an attribution grid. The grid keeps position, and head for (B, T, H, d_h) hooks."""
    if name.endswith("hook_pattern"):
        raise ValueError("patch hook_z instead of hook_pattern")
    return (0, 3) if a.dim() == 4 else (0, 2)


def activation_patch(model: GPT, clean: Tensor, corrupt: Tensor, name: str, metric: Metric, mode: str = "denoise",
                     positions: list[int] | None = None) -> Tensor:
    """Exact patching effect delta = m(patched) - m(unpatched run), one forward per cell.

    denoise: run corrupt, patch clean values in. noise: run clean, patch corrupt values in.
    Grid (T,) or (T, H) for per-head hooks. Rows outside `positions` stay 0.
    """
    _, cc = run_with_cache(model, clean, [name])
    _, cx = run_with_cache(model, corrupt, [name])
    src, run = (cc[name], corrupt) if mode == "denoise" else (cx[name], clean)
    with torch.no_grad():
        base = float(metric(model(run)))
    T, heads = src.shape[1], src.shape[2] if src.dim() == 4 else None
    out = torch.zeros(T, *(() if heads is None else (heads,)), dtype=torch.float64)
    for t in range(T) if positions is None else positions:
        for h in [None] if heads is None else range(heads):
            ix = (slice(None), t) if h is None else (slice(None), t, h)
            out[(t,) if h is None else (t, h)] = patch(model, run, name, src, metric, ix) - base
    return out


def attribution_patch(model: GPT, clean: Tensor, corrupt: Tensor, names: list[str], metric: Metric,
                      mode: str = "denoise") -> dict[str, Tensor]:
    """The same grids as activation_patch, from 2 forwards and 1 backward for all names."""
    _, other = run_with_cache(model, clean if mode == "denoise" else corrupt, names)
    _, acts, grads = _grad_run(model, corrupt if mode == "denoise" else clean, names, metric)
    return {n: (grads[n] * (other[n].to(acts[n].dtype) - acts[n])).sum(_grid_dims(acts[n], n)).double().cpu() for n in names}
