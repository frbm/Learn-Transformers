"""KV caches. Both follow one protocol, so GPT.forward has one code path:

    plan = cache.begin(T, **kw)     # positions and mask of the T new tokens
    K, V = cache.update(layer, k, v)  # once per layer
    cache.end(T)                    # commit
"""
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class AttnPlan:
    """Where the T new tokens sit. positions=None means q_offset, ..., q_offset + T - 1 in every row."""
    q_offset: int = 0
    positions: Tensor | None = None  # (B, T)
    mask: object = None  # None (causal at q_offset), a bool tensor, or a BlockMask


class KVCache:
    """Contiguous rows, one shared length. Pre-allocated once. Writes go in place at [pos, pos + T)."""

    def __init__(self, cfg, batch_size: int, device, dtype, n_slots: int | None = None):
        shape = (cfg.n_layers, batch_size, cfg.n_kv_heads, n_slots or cfg.max_seq_len, cfg.d_head)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.pos = 0

    def begin(self, T: int) -> AttnPlan:
        assert self.pos + T <= self.k.shape[3], "cache full"
        return AttnPlan(q_offset=self.pos)

    def update(self, layer: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        T = k.shape[2]
        self.k[layer, :, :, self.pos : self.pos + T] = k
        self.v[layer, :, :, self.pos : self.pos + T] = v
        return self.k[layer, :, :, : self.pos + T], self.v[layer, :, :, : self.pos + T]

    def end(self, T: int) -> None:
        self.pos += T


class BatchCache:
    """Left-padded rows of different lengths (D4). All rows share one write pointer `len`.

    valid[b, s] is True where slot s of row b holds a real token. n_real[b] is the next RoPE position of row b.
    Rollback only moves the pointer back. It never copies the cache.
    """

    def __init__(self, cfg, batch_size: int, n_slots: int, device, dtype):
        shape = (cfg.n_layers, batch_size, cfg.n_kv_heads, n_slots, cfg.d_head)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.valid = torch.zeros(batch_size, n_slots, dtype=torch.bool, device=device)
        self.n_real = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.n_slots, self.len = n_slots, 0

    def begin(self, T: int, valid: Tensor | None = None, slot: Tensor | None = None) -> AttnPlan:
        """valid (B, T): False on left padding. slot: a 1-element device tensor for a static-shape (CUDA graph) step.

        With slot, the step attends over all slots and does not move `len`.
        """
        dev, B = self.valid.device, self.valid.shape[0]
        plain_prefill = valid is None and slot is None and self.len == 0
        if valid is None:
            valid = torch.ones(B, T, dtype=torch.bool, device=dev)
        if slot is None:
            assert self.len + T <= self.n_slots, "cache full"
            self._slots, self._n_k = torch.arange(self.len, self.len + T, device=dev), self.len + T
        else:
            self._slots, self._n_k = slot + torch.arange(T, device=dev), self.n_slots
        self._valid, self._static = valid, slot is not None
        positions = (self.n_real[:, None] + valid.cumsum(1) - 1).clamp(min=0)
        self.valid.index_copy_(1, self._slots, valid)
        if plain_prefill:
            return AttnPlan(positions=positions)
        keys = torch.arange(self._n_k, device=dev)
        # A query sees valid earlier slots and always itself. Then a padding query has no empty softmax row.
        mask = (self.valid[:, None, : self._n_k] & (keys <= self._slots[:, None])) | (keys == self._slots[:, None])
        return AttnPlan(positions=positions, mask=mask[:, None])

    def update(self, layer: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        # index_copy_ takes a device tensor of slots. A CUDA graph can then replay the same step.
        self.k[layer].index_copy_(2, self._slots, k)
        self.v[layer].index_copy_(2, self._slots, v)
        return self.k[layer, :, :, : self._n_k], self.v[layer, :, :, : self._n_k]

    def end(self, T: int) -> None:
        self.n_real += self._valid.sum(1)
        if not self._static:
            self.len += T

    def rollback(self, keep: int) -> None:
        self.n_real -= self.valid[:, keep : self.len].sum(1)
        self.len = keep

    def reorder(self, rows: Tensor) -> None:
        """Beam search: row b takes the cache of row rows[b]."""
        self.k, self.v = self.k.index_select(1, rows), self.v.index_select(1, rows)
        self.valid, self.n_real = self.valid.index_select(0, rows), self.n_real.index_select(0, rows)
