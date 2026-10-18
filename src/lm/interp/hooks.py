"""E1 hook library. The HookPoints live in the model (lm.model.gpt), so there is no patch and no second forward.

Hook names (module paths in model.named_modules()):
  hook_embed, norm.hook_scale (rho of the final RMSNorm, (B, T, 1)),
  blocks.{l}.hook_resid_pre / hook_attn_out / hook_resid_mid / hook_mlp_out / hook_resid_post,
  blocks.{l}.attn.hook_q / hook_k / hook_v (after QK-norm and RoPE, (B, T, heads, d_h)),
  blocks.{l}.attn.hook_pattern (B, heads, T, T), blocks.{l}.attn.hook_z (B, T, heads, d_h).
"""
from contextlib import contextmanager
from typing import Callable

import torch
from torch import Tensor

from lm.model import GPT, Config, HookPoint


def hook_names(model: GPT, include_norms: bool = False) -> list[str]:
    """Every HookPoint. The RMS scales of inner norms are left out unless asked: they add L x 4 entries."""
    return [n for n, m in model.named_modules()
            if isinstance(m, HookPoint) and (include_norms or not n.endswith("hook_scale") or n == "norm.hook_scale")]


@contextmanager
def hooks(model: GPT, fwd: dict[str, Callable[[Tensor], Tensor | None]]):
    """Register forward hooks. A hook that returns a tensor replaces the activation. Hooks are always removed."""
    mods = dict(model.named_modules())
    handles = []
    try:
        for name, fn in fwd.items():
            handles.append(mods[name].register_forward_hook(lambda m, i, o, fn=fn: fn(o)))
        yield model
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def run_with_cache(model: GPT, idx: Tensor, names: list[str] | None = None, **fwd_kw) -> tuple[Tensor, dict[str, Tensor]]:
    names = hook_names(model) if names is None else names
    cache: dict[str, Tensor] = {}

    def store(n):
        def fn(o):
            cache[n] = o.detach()
        return fn

    with hooks(model, {n: store(n) for n in names}):
        logits = model(idx, **fwd_kw)
    return logits, cache


def cache_bytes(cfg: Config, B: int, T: int, act_bytes: int = 2, pattern_bytes: int = 4) -> dict[str, int]:
    """Memory of a full cache. Patterns are in fp32 (the work dtype), the rest in the model dtype."""
    btd = B * T * cfg.d_model
    kv = B * T * cfg.n_kv_heads * cfg.d_head
    resid = cfg.n_layers * (5 * btd + 2 * btd + 2 * kv) + btd + B * T  # 5 stream-sized, q and z, k and v
    pattern = cfg.n_layers * B * cfg.n_heads * T * T
    out = {"activations": resid * act_bytes, "patterns": pattern * pattern_bytes}
    out["total"] = sum(out.values())
    return out
