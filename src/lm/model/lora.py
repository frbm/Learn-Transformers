"""D2 Part A. LoRA, rsLoRA scaling, DoRA, PiSSA init, on nn.Linear or NF4Linear bases."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lm.model.quant import NF4Linear


def base_weight(base: nn.Module, dtype=None) -> Tensor:
    if isinstance(base, NF4Linear):
        return base.dequantized_weight(dtype or torch.float32)
    return base.weight if dtype is None else base.weight.to(dtype)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Module, r: int, alpha: float, scaling: str = "standard",
                 dropout: float = 0.0, init: str = "default", dora: bool = False):
        super().__init__()
        self.base, self.r, self.dora = base, r, dora
        for p in base.parameters():
            p.requires_grad_(False)
        d_in, d_out = base.in_features, base.out_features
        self.scale = alpha / r if scaling == "standard" else alpha / math.sqrt(r)
        ref = base.weight if isinstance(base, nn.Linear) else base.packed
        dt = ref.dtype if ref.is_floating_point() else torch.float32
        self.lora_A = nn.Parameter(torch.empty(r, d_in, device=ref.device, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r, device=ref.device, dtype=dt))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if init == "pissa":
            self._pissa()
        else:
            # The same init as the reference implementation. B = 0, so the layer starts equal to the base.
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        if dora:
            W0 = base_weight(base, dt).detach()
            self.lora_magnitude = nn.Parameter(W0.norm(dim=1))
            self.register_buffer("w0_sq", W0.pow(2).sum(1), persistent=False)

    @torch.no_grad()
    def _pissa(self) -> None:
        assert isinstance(self.base, nn.Linear), "apply PiSSA before quantisation, then quantise the residual"
        W0 = self.base.weight
        U, S, Vh = torch.linalg.svd(W0 if W0.dtype == torch.float64 else W0.float(), full_matrices=False)
        s = (S[: self.r] / self.scale).sqrt()
        self.lora_B.copy_(U[:, : self.r] * s)
        self.lora_A.copy_(s[:, None] * Vh[: self.r])
        # Eckart-Young: scale * B @ A is the best rank-r approximation. The base keeps the residual.
        W0.sub_(self.scale * self.lora_B @ self.lora_A)

    def delta(self) -> Tensor:
        return self.scale * self.lora_B @ self.lora_A

    def _row_norm(self) -> Tensor:
        """||W0_i + s (BA)_i|| without the (d_out, d_in) matrix. Detached, as in the DoRA paper."""
        with torch.no_grad():
            W0 = base_weight(self.base, self.lora_A.dtype)
            cross = ((W0 @ self.lora_A.T) * self.lora_B).sum(1)                       # <W0_i, (BA)_i>
            ba_sq = ((self.lora_B @ (self.lora_A @ self.lora_A.T)) * self.lora_B).sum(1)  # ||(BA)_i||^2
            return (self.w0_sq + 2 * self.scale * cross + self.scale**2 * ba_sq).clamp(min=0).sqrt()

    def forward(self, x: Tensor) -> Tensor:
        y = self.base(x)
        # Factored order: 2 r (d_in + d_out) FLOPs per token. Never form B @ A here.
        lora = self.scale * F.linear(F.linear(self.dropout(x), self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype))
        if not self.dora:
            return y + lora
        bias = self.base.bias
        if bias is not None:
            y = y - bias
        out = (self.lora_magnitude / self._row_norm()).to(y.dtype) * (y + lora)
        return out if bias is None else out + bias

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        W = base_weight(self.base, self.lora_A.dtype) + self.delta()
        if self.dora:
            W = (self.lora_magnitude / W.norm(dim=1))[:, None] * W
        out = nn.Linear(W.shape[1], W.shape[0], bias=self.base.bias is not None, device=W.device, dtype=W.dtype)
        out.weight.copy_(W)
        if self.base.bias is not None:
            out.bias.copy_(self.base.bias)
        return out


def apply_lora(model: nn.Module, targets: list[str], **kw) -> nn.Module:
    """Wrap every linear layer whose qualified name contains a target. Freeze everything else."""
    for p in model.parameters():
        p.requires_grad_(False)
    hits = [(n, m) for n, m in model.named_modules()
            if isinstance(m, (nn.Linear, NF4Linear)) and any(t in n for t in targets)]
    for name, mod in hits:
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent), leaf, LoRALinear(mod, **kw))
    return model


def lora_layers(model: nn.Module) -> dict[str, LoRALinear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}


def lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}


@torch.no_grad()
def merge_lora(model: nn.Module) -> nn.Module:
    for name, mod in lora_layers(model).items():
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent), leaf, mod.merge())
    return model
