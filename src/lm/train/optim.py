"""B2. Muon: steepest descent under the spectral norm for hidden matrices. AdamW for the rest."""
import math
from collections import defaultdict

import torch
from torch import Tensor

from lm.model.mup import classify, param_groups


def newton_schulz(G: Tensor, steps: int = 5, coeffs=(3.4445, -4.7750, 2.0315),
                  dtype=torch.bfloat16, eps: float = 1e-7) -> Tensor:
    """Approximate polar factor U V^T of G (..., m, n). Each step applies phi(s) = a s + b s^3 + c s^5 to every singular value."""
    a, b, c = coeffs
    X = G.to(dtype)
    tall = X.shape[-2] > X.shape[-1]
    if tall:
        X = X.mT                                     # Then X X^T is the smaller Gram matrix.
    # Frobenius >= spectral, so every singular value starts in (0, 1].
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    return X.mT if tall else X


@torch.no_grad()
def spectral_norm_estimate(W: Tensor, iters: int = 20, v: Tensor | None = None, block: int = 8) -> float:
    """Largest singular value by block power iteration on W^T W, then Rayleigh-Ritz. No SVD of W.

    One vector converges like (s2 / s1)^(2k), which is slow when s2 ~ s1 (Gaussian matrices). A block of b
    vectors converges like (s_{b+1} / s1)^(2k), at the cost of one (n, b) matmul per iteration.
    v: a warm start of shape (n, b). It is updated in place.
    """
    W = W.detach().float().flatten(1)
    if v is None:
        v = torch.randn(W.shape[1], block, device=W.device, generator=torch.Generator(W.device).manual_seed(0))
    if W.norm() == 0:
        # Keep the old start block. A zero block would make the next warm start 0 / 0.
        return 0.0
    X = torch.linalg.qr(v.float()).Q
    for _ in range(iters):
        X = torch.linalg.qr(W.mT @ (W @ X)).Q
    v.copy_(X)
    WX = W @ X
    return torch.linalg.eigvalsh(WX.mT @ WX)[-1].clamp_min(0).sqrt().item()


class Muon(torch.optim.Optimizer):
    """Momentum, Nesterov, Newton-Schulz, then W -= lr * sqrt(max(1, d_out / d_in)) * O. Only for 2-D tensors."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.0, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay,
                                      ns_steps=ns_steps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else torch.enable_grad()(closure)()
        for g in self.param_groups:
            by_shape: dict[tuple, list] = defaultdict(list)
            for p in g["params"]:
                if p.grad is None:
                    continue
                assert p.ndim == 2, "Muon is for matrices; give vectors to AdamW"
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buf = state["momentum_buffer"]
                buf.mul_(g["momentum"]).add_(p.grad)
                u = p.grad.add(buf, alpha=g["momentum"]) if g["nesterov"] else buf
                by_shape[tuple(p.shape)].append((p, u))
            # One Newton-Schulz call per shape, on a stack. At small d the step is launch-bound:
            # 24 matrices x 15 matmuls become 15 batched matmuls per shape.
            for (d_out, d_in), items in by_shape.items():
                O = newton_schulz(torch.stack([u for _, u in items]), steps=g["ns_steps"])
                scale = -g["lr"] * math.sqrt(max(1.0, d_out / d_in))
                for (p, _), o in zip(items, O):
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                    p.add_(o.to(p.dtype), alpha=scale)
        return loss


def adamw(model, lr: float, weight_decay: float, betas=(0.9, 0.95)) -> torch.optim.AdamW:
    """Decay on matrices only. Norm gains and other vectors get none."""
    decay = [p for p in model.parameters() if p.ndim >= 2]
    rest = [p for p in model.parameters() if p.ndim < 2]
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": rest, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=next(model.parameters()).is_cuda)


def build_optimizers(model, muon_lr, adam_lr, weight_decay, ns_steps: int = 5,
                     betas=(0.9, 0.95)) -> list[torch.optim.Optimizer]:
    """Muon for "hidden" tensors, AdamW with B1's per-kind groups for the rest."""
    groups = param_groups(model, adam_lr, weight_decay)
    hidden = [g for g in groups if g["kind"] == "hidden"]
    rest = [g for g in groups if g["kind"] != "hidden"]
    assert all(classify(n, p) == "hidden" for g in hidden for n, p in zip(g["names"], g["params"]))
    # μP for Muon is already in the sqrt(d_out / d_in) scale. So the Muon lr carries no width multiplier.
    muon = Muon([{"params": g["params"], "names": g["names"], "kind": "hidden", "lr_mult": 1.0} for g in hidden],
                lr=muon_lr, weight_decay=weight_decay, ns_steps=ns_steps)
    fused = next(model.parameters()).is_cuda
    return [muon, torch.optim.AdamW(rest, lr=adam_lr, betas=betas, fused=fused)]
