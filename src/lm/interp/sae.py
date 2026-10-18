"""E2 sparse autoencoders: proximal activations, the SAE, metrics, an activation buffer, steering."""
from contextlib import contextmanager
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lm.interp.hooks import hooks


def topk_activation(u: Tensor, k: int) -> Tensor:
    """Projection onto {z >= 0, ||z||_0 <= k}: keep the k largest entries, clip them at 0."""
    v, i = u.topk(k, dim=-1)
    return torch.zeros_like(u).scatter(-1, i, F.relu(v))


def soft_threshold(s: Tensor, lam: float) -> Tensor:
    """argmin_{z >= 0} (s - z)^2 + lam z."""
    return F.relu(s - lam / 2)


def hard_threshold(s: Tensor, lam: float) -> Tensor:
    """argmin_z (s - z)^2 + lam 1[z != 0]."""
    return s * (s * s > lam)


class _JumpReLU(torch.autograd.Function):
    """z = u 1[u > theta]. Straight-through pseudo-derivative for theta (Rajamanoharan et al. 2024)."""

    @staticmethod
    def forward(ctx, u, theta, eps):
        ctx.save_for_backward(u, theta)
        ctx.eps = eps
        return u * (u > theta)

    @staticmethod
    def backward(ctx, g):
        u, theta = ctx.saved_tensors
        kern = ((u - theta).abs() < ctx.eps / 2) / ctx.eps
        return g * (u > theta), (-theta * kern * g).flatten(0, -2).sum(0), None


class _Step(torch.autograd.Function):
    """1[u > theta], for the l0 penalty. Gradient only to theta."""

    @staticmethod
    def forward(ctx, u, theta, eps):
        ctx.save_for_backward(u, theta)
        ctx.eps = eps
        return (u > theta).to(u.dtype)

    @staticmethod
    def backward(ctx, g):
        u, theta = ctx.saved_tensors
        kern = ((u - theta).abs() < ctx.eps / 2) / ctx.eps
        return None, (-kern * g).flatten(0, -2).sum(0), None


class SAE(nn.Module):
    """z = sigma(W_enc (x - b_dec) + b_enc), x_hat = W_dec z + b_dec. kind: topk, relu (with l1), jumprelu (with l0)."""

    def __init__(self, d: int, m: int, k: int | None = None, kind: str = "topk", dtype=torch.float32, eps: float = 1e-3):
        super().__init__()
        assert kind != "topk" or k is not None, "TopK needs k"
        self.d, self.m, self.k, self.kind, self.eps = d, m, k, kind, eps
        W = torch.randn(d, m, dtype=dtype)
        W /= W.norm(dim=0, keepdim=True)
        self.W_dec = nn.Parameter(W)
        # W_enc = W_dec^T: the orthonormal case of Proposition 5 is the start point. It prevents most dead latents.
        self.W_enc = nn.Parameter(W.T.clone())
        self.b_enc = nn.Parameter(torch.zeros(m, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d, dtype=dtype))
        if kind == "jumprelu":
            self.log_theta = nn.Parameter(torch.full((m,), -3.0, dtype=dtype))

    def pre(self, x: Tensor) -> Tensor:
        return (x - self.b_dec) @ self.W_enc.T + self.b_enc

    def activate(self, u: Tensor) -> Tensor:
        if self.kind == "topk":
            return topk_activation(u, self.k)
        if self.kind == "relu":
            return F.relu(u)
        return _JumpReLU.apply(u, self.log_theta.exp(), self.eps)

    def encode(self, x: Tensor) -> Tensor:
        return self.activate(self.pre(x))

    def decode(self, z: Tensor) -> Tensor:
        return z @ self.W_dec.T + self.b_dec

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = self.encode(x)
        return self.decode(z), z

    def sparsity_penalty(self, x: Tensor, z: Tensor) -> Tensor:
        """l1 for relu (scaled by decoder norms), l0 through the step pseudo-derivative for jumprelu, 0 for topk."""
        if self.kind == "relu":
            return (z * self.W_dec.norm(dim=0)).sum(-1).mean()
        if self.kind == "jumprelu":
            return _Step.apply(self.pre(x), self.log_theta.exp(), self.eps).sum(-1).mean()
        return z.new_zeros(())


def aux_k_loss(sae: SAE, x: Tensor, x_hat: Tensor, dead: Tensor, k_aux: int = 512) -> Tensor:
    """AuxK (Gao et al. 2024): the dead latents reconstruct the residual error. No decoder bias."""
    n_dead = int(dead.sum())
    if n_dead == 0:
        return x.new_zeros(())
    e = (x - x_hat).detach()
    u = sae.pre(x).masked_fill(~dead, float("-inf"))
    e_hat = topk_activation(u, min(k_aux, n_dead)) @ sae.W_dec.T
    return ((e - e_hat) ** 2).sum(-1).mean() / (e**2).sum(-1).mean().clamp_min(1e-12)


@torch.no_grad()
def normalize_decoder_(sae: SAE) -> None:
    sae.W_dec /= sae.W_dec.norm(dim=0, keepdim=True)


@torch.no_grad()
def remove_parallel_grad_(sae: SAE) -> None:
    """g <- g - <g, w> w / ||w||^2 per column: the step stays tangent to the sphere of unit columns."""
    w, g = sae.W_dec, sae.W_dec.grad
    if g is not None:
        g -= (g * w).sum(0, keepdim=True) / (w * w).sum(0, keepdim=True) * w


def fvu(x: Tensor, x_hat: Tensor) -> Tensor:
    return ((x - x_hat) ** 2).sum() / ((x - x.mean(0)) ** 2).sum()


def loss_recovered(ce_clean: float, ce_sae: float, ce_zero: float) -> float:
    return (ce_zero - ce_sae) / (ce_zero - ce_clean)


class _Stop(Exception):
    pass


class ActivationBuffer:
    """A shuffled buffer of activations at one hook, streamed from forward passes.

    The first fill fills every slot. When half the slots are consumed, new activations replace them and the order
    is shuffled again. Thus a batch mixes many documents. Position 0 and end-of-text positions are dropped: their
    norms are many times the rest (attention sink). x * scale has E||x||^2 = d. Divide by scale to splice back.
    """

    def __init__(self, model, name: str, tokens: Callable[[], Tensor], size: int = 2**20, eot: int | None = None,
                 store_dtype=torch.bfloat16, seed: int = 0, autocast: bool = True):
        self.model, self.name, self.tokens, self.size, self.eot, self.autocast = model, name, tokens, size, eot, autocast
        p = next(model.parameters())
        self.device, self.d = p.device, model.cfg.d_model
        self.g = torch.Generator(device=self.device).manual_seed(seed)
        self.buf = torch.empty(size, self.d, dtype=store_dtype, device=self.device)  # preallocated once
        self.n_tokens = 0
        self._fill(torch.arange(size, device=self.device))
        x = self.buf[: min(size, 65536)].float()
        self.scale = (self.d / x.pow(2).sum(-1).mean()).sqrt().item()
        self._shuffle()

    @torch.no_grad()
    def activations(self, idx: Tensor) -> Tensor:
        """(N, d) activations of the batch, exclusions applied. Stop the forward pass after the hook layer."""
        out = {}

        def grab(o):
            out["a"] = o
            raise _Stop

        with hooks(self.model, {self.name: grab}), torch.autocast(self.device.type, torch.bfloat16, enabled=self.autocast):
            try:
                self.model(idx)
            except _Stop:
                pass
        keep = torch.ones_like(idx, dtype=torch.bool)
        keep[:, 0] = False
        if self.eot is not None:
            keep &= idx != self.eot
        self.n_tokens += idx.numel()
        return out["a"][keep]

    def _fill(self, slots: Tensor) -> None:
        i = 0
        while i < len(slots):
            a = self.activations(self.tokens())
            n = min(len(a), len(slots) - i)
            self.buf[slots[i : i + n]] = a[:n].to(self.buf.dtype)
            i += n

    def _shuffle(self) -> None:
        self.perm = torch.randperm(self.size, generator=self.g, device=self.device)
        self.ptr = 0

    def next(self, n: int) -> Tensor:
        if self.ptr + n > self.size // 2:
            self._fill(self.perm[: self.ptr])
            self._shuffle()
        out = self.buf[self.perm[self.ptr : self.ptr + n]].float() * self.scale
        self.ptr += n
        return out


@contextmanager
def steer(model, layer: int, v: Tensor, alpha: float):
    """Add alpha * v to blocks.{layer}.hook_resid_post at every position."""
    with hooks(model, {f"blocks.{layer}.hook_resid_post": lambda o: o + (alpha * v).to(o.dtype)}):
        yield model
