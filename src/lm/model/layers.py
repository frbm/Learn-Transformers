import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class HookPoint(nn.Module):
    """Identity module. Forward hooks on it read or replace one activation (E1). It costs nothing without hooks."""

    def forward(self, x: Tensor) -> Tensor:
        return x


def work_dtype(dt: torch.dtype) -> torch.dtype:
    """fp32 for bf16, fp16 and fp32. fp64 for fp64, so that float64 tests stay exact."""
    return torch.promote_types(dt, torch.float32)


def rms_scale(x: Tensor, eps: float) -> Tensor:
    """sqrt(mean(x^2) + eps) over the last axis, in the work dtype. Shape (..., 1)."""
    xf = x.to(work_dtype(x.dtype))
    return torch.sqrt(xf.pow(2).mean(-1, keepdim=True) + eps)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        # The RMS itself, shape (..., 1). A patch here freezes the norm (E1 direct logit attribution).
        self.hook_scale = HookPoint()

    def forward(self, x: Tensor) -> Tensor:
        rho = self.hook_scale(rms_scale(x, self.eps))
        return (x.to(rho.dtype) / rho).to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
