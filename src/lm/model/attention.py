"""Attention kernels and masks.

A mask is one of:
- None: causal, with query i at absolute position q_offset + i.
- a bool tensor that broadcasts to (B, n_heads, T_q, T_k), True where attention is allowed.
- a flex-attention BlockMask (packed documents on a GPU with Triton).
"""
import functools
import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

from lm.model.layers import work_dtype


def causal_mask(T_q: int, T_k: int, q_offset: int, device) -> Tensor:
    return torch.ones(T_q, T_k, dtype=torch.bool, device=device).tril(diagonal=q_offset)


def doc_causal_mask(doc_ids: Tensor) -> Tensor:
    """(B, 1, T, T): causal inside each document of a packed row.

    doc_ids < 0 marks padding. A padding query sees only itself, so that no softmax row is empty.
    """
    T = doc_ids.shape[1]
    same = doc_ids[:, :, None] == doc_ids[:, None, :]
    causal = torch.ones(T, T, dtype=torch.bool, device=doc_ids.device).tril()
    eye = torch.eye(T, dtype=torch.bool, device=doc_ids.device)
    return ((same & causal & (doc_ids[:, :, None] >= 0)) | eye)[:, None]


def attention_weights(q: Tensor, k: Tensor, q_offset: int = 0, scale: float | None = None,
                      mask: Tensor | None = None) -> Tensor:
    """Softmax weights in the work dtype, shape (B, n_kv, g, T_q, T_k). Query head j uses KV head j // g."""
    B, n_q, T_q, d_h = q.shape
    n_kv, T_k = k.shape[1], k.shape[2]
    scale = 1 / math.sqrt(d_h) if scale is None else scale
    dt = work_dtype(q.dtype)
    # Broadcast over the group axis. This shares K and V without a copy.
    s = (q.view(B, n_kv, n_q // n_kv, T_q, d_h).to(dt) @ k.unsqueeze(2).to(dt).transpose(-1, -2)) * scale
    allowed = causal_mask(T_q, T_k, q_offset, q.device) if mask is None else mask.unsqueeze(2)
    return s.masked_fill(~allowed, float("-inf")).softmax(dim=-1)


def attention_core(q: Tensor, k: Tensor, v: Tensor, q_offset: int = 0, scale: float | None = None,
                   mask: Tensor | None = None) -> Tensor:
    """The definition: scores, mask, softmax in fp32, weighted sum. The oracle for every fast path."""
    B, n_q, T_q, d_h = q.shape
    p = attention_weights(q, k, q_offset, scale, mask)
    return (p.to(v.dtype) @ v.unsqueeze(2)).view(B, n_q, T_q, d_h)


# The flash kernel is missing from the Windows wheels, and the efficient kernel rejects enable_gqa.
# Without a priority list, SDPA then picks the math kernel: it stores (B, h, T, T) scores and is 13x slower.
# cuDNN takes GQA natively and saves only O and the log-sum-exp.
_FUSED = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


def attention_sdpa(q: Tensor, k: Tensor, v: Tensor, q_offset: int = 0, scale: float | None = None,
                   mask: Tensor | None = None) -> Tensor:
    if mask is None and q_offset == 0 and q.shape[2] == k.shape[2]:
        if q.is_cuda:
            with sdpa_kernel(_FUSED, set_priority=True):
                return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale, enable_gqa=True)
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale, enable_gqa=True)
    # is_causal=True aligns the mask top-left when T_q != T_k. Give the mask explicitly.
    if mask is None:
        mask = causal_mask(q.shape[2], k.shape[2], q_offset, q.device)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale, enable_gqa=True)


@functools.cache
def _flex():
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    # Uncompiled, flex_attention runs a slow reference implementation.
    return torch.compile(flex_attention, dynamic=False), create_block_mask


@functools.cache
def flex_available() -> bool:
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


def flex_doc_mask(doc_ids: Tensor):
    """The BlockMask of doc_causal_mask. Fully masked tiles are skipped, not computed."""
    _, create_block_mask = _flex()

    def doc_causal(b, h, q, kv):
        d = doc_ids[b, q]
        return ((d == doc_ids[b, kv]) & (d >= 0) & (q >= kv)) | (q == kv)

    B, T = doc_ids.shape
    return create_block_mask(doc_causal, B, None, T, T, device=doc_ids.device)


def is_block_mask(mask) -> bool:
    return mask is not None and not isinstance(mask, Tensor)


def attention_flex(q: Tensor, k: Tensor, v: Tensor, block_mask, scale: float | None = None) -> Tensor:
    return _flex()[0](q, k, v, block_mask=block_mask, scale=scale, enable_gqa=True)
