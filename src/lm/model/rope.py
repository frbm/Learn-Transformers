"""Rotary position embeddings.

Pairing convention: half-split (GPT-NeoX). Plane j is the coordinate pair (x[j], x[j + d_h/2]).
"""
import math

import torch
from torch import Tensor


def rope_freqs(d_h: int, base: float, scaling: dict | None = None) -> Tensor:
    theta = base ** (-torch.arange(0, d_h, 2, dtype=torch.float64) / d_h)
    if not scaling:
        return theta
    s = float(scaling["factor"])
    kind = scaling["type"]
    if kind == "pi":
        return theta / s
    if kind == "ntk":
        return (base * s ** (d_h / (d_h - 2))) ** (-torch.arange(0, d_h, 2, dtype=torch.float64) / d_h)
    if kind == "yarn":
        turns = scaling["orig_ctx"] * theta / (2 * math.pi)
        a, b = scaling.get("alpha", 1), scaling.get("beta", 32)
        gamma = ((turns - a) / (b - a)).clamp(0, 1)
        return (1 - gamma) * theta / s + gamma * theta
    raise ValueError(f"unknown rope scaling {kind!r}")


def yarn_mscale(scaling: dict | None) -> float:
    """Factor on q and on k. The logits get its square."""
    if not scaling or scaling["type"] != "yarn" or scaling["factor"] <= 1:
        return 1.0
    return 0.1 * math.log(scaling["factor"]) + 1.0


def rope_angles(d_h: int, max_pos: int, base: float, scaling: dict | None = None) -> tuple[Tensor, Tensor]:
    # Build m * theta in float64. In fp32, angles near 1e5 rad have errors near 1e-2 rad.
    ang = torch.outer(torch.arange(max_pos, dtype=torch.float64), rope_freqs(d_h, base, scaling))
    return ang.cos().float(), ang.sin().float()


def rotate(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate each plane of x by the given angles. cos and sin broadcast against x[..., : d_h / 2]."""
    h = x.shape[-1] // 2
    c, s = cos.to(x.dtype), sin.to(x.dtype)
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor, offset: int = 0) -> Tensor:
    """x: (B, H, T, d_h) at absolute positions offset, ..., offset + T - 1."""
    T = x.shape[-2]
    return rotate(x, cos[offset : offset + T], sin[offset : offset + T])
