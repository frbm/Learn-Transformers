"""D2 Part A. Blockwise absmax 4-bit quantisation, the NF4 codebook, double quantisation, a frozen NF4 linear layer."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

NF4_OFFSET = 0.5 * ((1 - 1 / 30) + (1 - 1 / 32))


def nf4_codebook() -> Tensor:
    """8 positive and 7 negative Gaussian quantiles, an exact 0, scaled to [-1, 1] (QLoRA appendix)."""
    ndtri = lambda p: torch.special.ndtri(p)
    pos = ndtri(torch.linspace(NF4_OFFSET, 0.5, 9, dtype=torch.float64)[:-1])
    neg = -ndtri(torch.linspace(NF4_OFFSET, 0.5, 8, dtype=torch.float64)[:-1])
    v = torch.cat([pos, torch.zeros(1, dtype=torch.float64), neg]).sort().values
    return (v / v.abs().max()).float()


def uniform_codebook(n: int = 16) -> Tensor:
    return torch.linspace(-1, 1, n)


def nearest_index(u: Tensor, code: Tensor) -> Tensor:
    """Index of the nearest codebook point. bucketize on the midpoints: no (n, 16) distance matrix."""
    mids = (code[1:] + code[:-1]) / 2
    return torch.bucketize(u, mids.to(u.device, u.dtype)).to(torch.uint8)


def pack4(idx: Tensor) -> Tensor:
    return (idx[0::2] << 4) | idx[1::2]


def unpack4(packed: Tensor) -> Tensor:
    return torch.stack([packed >> 4, packed & 0xF], -1).flatten()


def quantize_blockwise(w: Tensor, code: Tensor, blocksize: int = 64) -> tuple[Tensor, Tensor]:
    assert w.numel() % blocksize == 0 and blocksize % 2 == 0, "pad the tensor to a multiple of the block size"
    blocks = w.detach().reshape(-1, blocksize).float()
    absmax = blocks.abs().amax(1)
    # An all-zero block has absmax 0. Divide by 1 instead. The codebook has an exact 0.
    u = blocks / torch.where(absmax > 0, absmax, 1.0)[:, None]
    return pack4(nearest_index(u.flatten(), code)), absmax


def dequantize_blockwise(packed: Tensor, absmax: Tensor, code: Tensor, shape, blocksize: int = 64,
                         dtype=torch.bfloat16) -> Tensor:
    vals = code.to(packed.device)[unpack4(packed).long()].view(-1, blocksize) * absmax[:, None].float()
    return vals.view(shape).to(dtype)


def double_quantize(absmax: Tensor, blocksize: int = 256) -> tuple[Tensor, Tensor, float]:
    offset = absmax.float().mean().item()
    a = absmax.float() - offset
    a = F.pad(a, (0, -len(a) % blocksize)).view(-1, blocksize)
    scale2 = a.abs().amax(1).clamp(min=torch.finfo(torch.float32).tiny)
    q = (a / scale2[:, None] * 127).round().add(128).to(torch.uint8)
    return q.flatten(), scale2, offset


def dequantize_absmax(q: Tensor, scale2: Tensor, offset: float, blocksize: int = 256) -> Tensor:
    return ((q.view(-1, blocksize).float() - 128) / 127 * scale2[:, None]).flatten() + offset


def bits_per_param(blocksize: int = 64, double_quant: bool = True, blocksize2: int = 256) -> float:
    return 4 + (8 / blocksize + 32 / (blocksize * blocksize2) if double_quant else 32 / blocksize)


def code_entropy_bits(packed: Tensor) -> float:
    """Empirical entropy of the 4-bit indices (Proposition 7: at most 4 bits)."""
    p = torch.bincount(unpack4(packed).long(), minlength=16).double()
    p = p[p > 0] / p.sum()
    return -(p * p.log2()).sum().item()


class NF4Linear(nn.Module):
    def __init__(self, linear: nn.Linear, blocksize: int = 64, double_quant: bool = True, code: Tensor | None = None):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.blocksize, self.double_quant = blocksize, double_quant
        dev = linear.weight.device
        self.register_buffer("code", (nf4_codebook() if code is None else code).to(dev))
        packed, absmax = quantize_blockwise(linear.weight, self.code, blocksize)
        self.register_buffer("packed", packed)
        self.n_blocks = len(absmax)
        if double_quant:
            q, scale2, self.offset = double_quantize(absmax)
            self.register_buffer("absmax_q", q)
            self.register_buffer("scale2", scale2)
        else:
            self.register_buffer("absmax", absmax)
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.detach().clone(), requires_grad=False)

    def block_scales(self) -> Tensor:
        if self.double_quant:
            return dequantize_absmax(self.absmax_q, self.scale2, self.offset)[: self.n_blocks]
        return self.absmax

    def dequantized_weight(self, dtype) -> Tensor:
        return dequantize_blockwise(self.packed, self.block_scales(), self.code, (self.out_features, self.in_features),
                                    self.blocksize, dtype)

    def forward(self, x: Tensor) -> Tensor:
        dtype = torch.get_autocast_dtype("cuda") if x.is_cuda and torch.is_autocast_enabled("cuda") else x.dtype
        return F.linear(x, self.dequantized_weight(dtype), self.bias)
