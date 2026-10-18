"""The decoder: pre-norm blocks, GQA with QK-norm and RoPE, SwiGLU, tied embeddings.

One forward serves every module: training (optionally checkpointed), packed documents (doc_ids, position_ids),
KV-cached decoding (KVCache, BatchCache) and interpretability (HookPoints, see lm.interp.hooks).
"""
import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from lm.model.attention import (attention_core, attention_flex, attention_sdpa, attention_weights, doc_causal_mask,
                                flex_available, flex_doc_mask, is_block_mask)
from lm.model.cache import AttnPlan, BatchCache, KVCache
from lm.model.config import Config
from lm.model.layers import HookPoint, RMSNorm, SwiGLU
from lm.model.loss import cross_entropy_from_hidden
from lm.model.rope import rope_angles, rotate, yarn_mscale


class Attention(nn.Module):
    def __init__(self, cfg: Config, layer: int):
        super().__init__()
        self.cfg, self.layer = cfg, layer
        d, dh = cfg.d_model, cfg.d_head
        self.n_q, self.n_kv = cfg.n_heads, cfg.n_kv_heads
        self.wq = nn.Linear(d, self.n_q * dh, bias=False)
        self.wk = nn.Linear(d, self.n_kv * dh, bias=False)
        self.wv = nn.Linear(d, self.n_kv * dh, bias=False)
        self.wo = nn.Linear(self.n_q * dh, d, bias=False)
        self.q_norm = RMSNorm(dh, cfg.norm_eps)
        self.k_norm = RMSNorm(dh, cfg.norm_eps)
        self.scale = yarn_mscale(cfg.rope_scaling) ** 2 / math.sqrt(dh)
        self.core = attention_core if cfg.attn_impl == "naive" else attention_sdpa
        # Hook layout: (B, T, heads, d_h) for q, k, v, z; (B, heads, T_q, T_k) for the pattern.
        for name in ("q", "k", "v", "pattern", "z"):
            self.add_module(f"hook_{name}", HookPoint())

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor] | None, plan: AttnPlan, cache=None) -> Tensor:
        B, T, _ = x.shape
        dh = self.cfg.d_head
        q = self.q_norm(self.wq(x).view(B, T, self.n_q, dh))
        k = self.k_norm(self.wk(x).view(B, T, self.n_kv, dh))
        v = self.wv(x).view(B, T, self.n_kv, dh)
        # Normalise, then rotate. A rotation keeps the norm, so the QK-norm logit bound stays.
        if rope is not None:
            q, k = rotate(q, *rope), rotate(k, *rope)
        # The fp32 norm gains promote q and k under autocast. Match v: flex attention is not autocast-aware.
        q, k = q.to(v.dtype), k.to(v.dtype)
        q = self.hook_q(q).transpose(1, 2)
        k = self.hook_k(k).transpose(1, 2)
        v = self.hook_v(v).transpose(1, 2)
        if cache is not None:
            k, v = cache.update(self.layer, k, v)
        y = self._attend(q, k, v, plan)
        return self.wo(self.hook_z(y.transpose(1, 2)).reshape(B, T, self.n_q * dh))

    def _attend(self, q: Tensor, k: Tensor, v: Tensor, plan: AttnPlan) -> Tensor:
        if self.hook_pattern._forward_hooks:
            # Build the pattern only when a hook reads or writes it. Otherwise the fused kernel runs.
            B, n_q, T_q, dh = q.shape
            p = attention_weights(q, k, plan.q_offset, self.scale, plan.mask).view(B, n_q, T_q, -1)
            p = self.hook_pattern(p).view(B, self.n_kv, n_q // self.n_kv, T_q, -1)
            return (p.to(v.dtype) @ v.unsqueeze(2)).view(B, n_q, T_q, dh)
        if is_block_mask(plan.mask):
            return attention_flex(q, k, v, plan.mask, self.scale)
        return self.core(q, k, v, q_offset=plan.q_offset, scale=self.scale, mask=plan.mask)


class Block(nn.Module):
    def __init__(self, cfg: Config, layer: int):
        super().__init__()
        self.cfg = cfg
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg, layer)
        if cfg.use_mlp:
            self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
            self.mlp = SwiGLU(cfg.d_model, cfg.mlp_hidden)
        for name in ("resid_pre", "attn_out", "resid_mid", "mlp_out", "resid_post"):
            self.add_module(f"hook_{name}", HookPoint())

    def forward(self, x: Tensor, rope=None, plan: AttnPlan | None = None, cache=None) -> Tensor:
        x = self.hook_resid_pre(x)
        a = self.hook_attn_out(self.attn(self.attn_norm(x), rope, plan or AttnPlan(), cache))
        x = self.hook_resid_mid(x + a if self.cfg.use_residual else a)
        if self.cfg.use_mlp:
            m = self.hook_mlp_out(self.mlp(self.mlp_norm(x)))
            x = x + m if self.cfg.use_residual else m
        return self.hook_resid_post(x)


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.hook_embed = HookPoint()
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        if cfg.pos_encoding == "rope":
            cos, sin = rope_angles(cfg.d_head, cfg.max_seq_len, cfg.rope_base, cfg.rope_scaling)
            # Non-persistent: the tables follow .to(device) but stay out of checkpoints.
            # A checkpoint then loads under a new max_seq_len or rope_scaling (A4).
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)

    def _init(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=self.cfg.init_std)

    @property
    def device(self) -> torch.device:
        return self.tok_emb.weight.device

    @property
    def rope(self) -> tuple[Tensor, Tensor] | None:
        return (self.rope_cos, self.rope_sin) if self.cfg.pos_encoding == "rope" else None

    def _rope_for(self, plan: AttnPlan, T: int) -> tuple[Tensor, Tensor] | None:
        if self.cfg.pos_encoding != "rope":
            return None
        if plan.positions is None:
            cos, sin = self.rope_cos[plan.q_offset : plan.q_offset + T], self.rope_sin[plan.q_offset : plan.q_offset + T]
            return cos[:, None], sin[:, None]  # (T, 1, d_h / 2): broadcast over (B, T, heads, d_h / 2)
        return self.rope_cos[plan.positions][:, :, None], self.rope_sin[plan.positions][:, :, None]

    def _plan(self, idx: Tensor, cache, position_ids, doc_ids, flex, cache_kw) -> AttnPlan:
        T = idx.shape[1]
        if cache is not None:
            assert position_ids is None and doc_ids is None, "the cached path takes positions from the cache"
            return cache.begin(T, **cache_kw)
        if doc_ids is None and position_ids is None:
            assert T <= self.cfg.max_seq_len, "sequence longer than max_seq_len"
            return AttnPlan()
        if position_ids is None:
            position_ids = torch.arange(T, device=idx.device).expand_as(idx)
        mask = None
        if doc_ids is not None:
            # Build the mask once per batch. All layers share it.
            use_flex = (doc_ids.is_cuda and flex_available()) if flex is None else flex
            mask = flex_doc_mask(doc_ids) if use_flex else doc_causal_mask(doc_ids)
        return AttnPlan(positions=position_ids, mask=mask)

    def hidden(self, idx: Tensor, cache: KVCache | BatchCache | None = None, *, position_ids: Tensor | None = None,
               doc_ids: Tensor | None = None, flex: bool | None = None, **cache_kw) -> Tensor:
        """Final residual stream before the last norm. (B, T) -> (B, T, d)."""
        T = idx.shape[1]
        plan = self._plan(idx, cache, position_ids, doc_ids, flex, cache_kw)
        rope = self._rope_for(plan, T)
        x = self.hook_embed(self.tok_emb(idx))
        ckpt = self.cfg.checkpointing and cache is None and torch.is_grad_enabled()
        for b in self.blocks:
            # Checkpointing keeps only the block input. Backward recomputes the block.
            x = checkpoint(b, x, rope, plan, use_reentrant=False) if ckpt else b(x, rope, plan, cache)
        if cache is not None:
            cache.end(T)
        return x

    def logits(self, h: Tensor) -> Tensor:
        return self.lm_head(self.norm(h))

    def forward(self, idx: Tensor, cache=None, *, last_only: bool = False, **kw) -> Tensor:
        h = self.hidden(idx, cache, **kw)
        # last_only: a prefill of B x T x V logits does not fit at large B. Keep only what the caller needs.
        return self.logits(h[:, -1:] if last_only else h)

    def loss(self, idx: Tensor, targets: Tensor, reduction: str = "mean", ce_chunk: int | None = None, **kw) -> Tensor:
        """Next-token loss over targets != -100. Mean is a ratio of sums, never a mean of per-row means."""
        h = self.norm(self.hidden(idx, **kw))
        return cross_entropy_from_hidden(h.flatten(0, 1), self.lm_head.weight, targets.flatten(), ce_chunk, reduction)

    def init_cache(self, batch_size: int, dtype=None) -> KVCache:
        p = self.tok_emb.weight
        return KVCache(self.cfg, batch_size, p.device, dtype or p.dtype)

    @torch.no_grad()
    def generate(self, prompt: Tensor, n_new: int, use_cache: bool = True) -> Tensor:
        """Greedy. The cached path must give the same tokens as the uncached one (A3 test)."""
        x = prompt
        if not use_cache:
            for _ in range(n_new):
                x = torch.cat([x, self(x)[:, -1].argmax(-1, keepdim=True)], dim=1)
            return x
        cache = self.init_cache(prompt.shape[0])
        logits = self(prompt, cache, last_only=True)
        out = [prompt]
        for i in range(n_new):
            out.append(logits[:, -1].argmax(-1, keepdim=True))
            if i < n_new - 1:
                logits = self(out[-1], cache)
        return torch.cat(out, dim=1)
