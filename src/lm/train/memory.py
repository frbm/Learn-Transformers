"""C1 Section 1. Bytes of one training step: static state, saved activations, logits.

The coefficients count what this eager model saves under bf16 autocast, op by op (docs/C1.md, Task 2).
They are larger than Proposition 2's fused lower bound (32 + 4g) d.
"""

import torch

from lm.model import expected_params
WORKSPACE = 21 * 2**20


def _n_matrix(cfg) -> int:
    """Parameters of the hidden matrices (Muon's tensors): all block weights except the norm gains."""
    d, dh, h = cfg.d_model, cfg.d_head, cfg.mlp_hidden
    attn = 2 * d * cfg.n_heads * dh + 2 * d * cfg.n_kv_heads * dh
    return cfg.n_layers * (attn + (3 * d * h if cfg.use_mlp else 0))


def static_bytes(cfg, optimizer: str = "adamw") -> int:
    """Proposition 1: fp32 weights (4) + fp32 grads (4) + optimizer state + the bf16 autocast copies (2)."""
    N = expected_params(cfg)
    if optimizer == "adamw":
        return 18 * N
    if optimizer == "muon":
        return 18 * N - 4 * _n_matrix(cfg)
    raise ValueError(f"unknown optimizer {optimizer!r}")


def layer_bytes_per_token(cfg) -> int:
    """Bytes that one eager block saves for backward, per token.

    Attention: input x (4d) and normed x (4d) of RMSNorm; three bf16 casts of the normed x for wq, wk, wv (6d);
    q_norm and k_norm each save the fp32 upcast and the bf16 normed tensor (6d + 6kv); v (2kv);
    the bf16 casts of q and k for the kernel (2d + 2kv); the kernel output and its contiguous copy for wo (4d).
    MLP: x and normed x (8d); two bf16 casts for w1 and w3 (4d); a, b, silu(a), silu(a) * b in bf16 (8h).
    """
    d, kv, h = cfg.d_model, cfg.n_kv_heads * cfg.d_head, cfg.mlp_hidden
    attn = 26 * d + 10 * kv
    return attn + (12 * d + 8 * h if cfg.use_mlp else 0)


def activation_bytes(cfg, B: int, T: int, checkpointing: bool = False) -> int:
    """Saved activations at their peak, without the readout and the loss.

    Without checkpointing: all L layers plus the final norm's input (4d) and its pre-gain output (4d).
    With per-block checkpointing: the L + 1 fp32 block boundaries, the final norm, and one recomputed layer.
    """
    d, M = cfg.d_model, B * T
    final = 8 * d + 8  # final norm, plus the int64 token ids saved by the embedding.
    if checkpointing:
        return M * (cfg.n_layers * 4 * d + final + layer_bytes_per_token(cfg))
    return M * (cfg.n_layers * layer_bytes_per_token(cfg) + final)


def logits_bytes(cfg, B: int, T: int, ce_chunk: int | None = None) -> int:
    """Peak bytes of the readout and the loss, at the start of backward.

    Per logit: the saved fp32 log-softmax (4), its incoming gradient (4) and the outgoing gradient (4).
    The bf16 logits and their fp32 upcast are freed before backward. So 12 bytes per logit.
    Unchunked, the readout saves its bf16 input (2d). Chunked, the chunks save slices of the fp32 h (4d), and
    the backward of each slice accumulates into a full fp32 gradient of h, with one temporary (2 x 4d).
    """
    M, V, d = B * T, cfg.vocab_size, cfg.d_model
    if ce_chunk is None or ce_chunk >= M:
        return 12 * M * V + 2 * M * d
    return 12 * ce_chunk * V + 12 * M * d + 2 * V * d  # last term: the bf16 copy of W in the recompute


def predict_peak_memory(cfg, B: int, T: int, optimizer: str = "adamw", checkpointing: bool = False,
                        ce_chunk: int | None = None) -> int:
    """Peak of memory_allocated in a steady-state step (optimizer state exists). The max over three moments."""
    N, M, d, V = expected_params(cfg), B * T, cfg.d_model, cfg.vocab_size
    weights_and_state = static_bytes(cfg, optimizer) - 6 * N  # no grads and no bf16 copies yet
    copies = 0 if checkpointing else 2 * N  # checkpointing frees the copies with the autocast cache
    layer = layer_bytes_per_token(cfg) * M
    saved = activation_bytes(cfg, B, T, checkpointing) - (layer if checkpointing else 0)
    readout_grad = 4 * V * d
    # 1. Start of backward: everything saved, plus the logits gradients.
    ce = weights_and_state + copies + saved + logits_bytes(cfg, B, T, ce_chunk)
    # 2. Backward of the last block: one recomputed layer, one bf16 hidden-width gradient, its weight grads.
    layer_grads = 4 * (expected_params(cfg) - V * d) // cfg.n_layers
    last = weights_and_state + copies + saved + readout_grad + layer_grads + 2 * cfg.mlp_hidden * M
    if checkpointing:
        last += layer
    # 3. Optimizer step: weights, grads, state.
    opt = static_bytes(cfg, optimizer) - 2 * N
    return max(ce, last, opt) + WORKSPACE


def measure_peak_memory(cfg, B: int, T: int, checkpointing: bool = False, ce_chunk: int | None = None,
                        snapshot: str | None = None) -> int:
    """The test's protocol: two AdamW steps, peak of the second. Optionally dump a memory snapshot of it."""
    from lm.model import GPT

    model = GPT(cfg).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    idx = torch.randint(0, cfg.vocab_size, (B, T), device="cuda")
    for step in range(2):
        if step == 1 and snapshot:
            torch.cuda.memory._record_memory_history(max_entries=100_000)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model.loss(idx, idx, ce_chunk=ce_chunk)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step == 0:
            torch.cuda.reset_peak_memory_stats()
    peak = torch.cuda.max_memory_allocated()
    if snapshot:
        torch.cuda.memory._dump_snapshot(snapshot)
        torch.cuda.memory._record_memory_history(enabled=None)
    del model, opt, loss
    torch.cuda.empty_cache()
    return peak


def roofline_step_time(cfg, B: int, T: int, K: int, P: float, beta: float) -> float:
    """Lower bound of the step time: matmul FLOPs at P, plus the activation traffic at beta.

    Crude traffic model: each saved activation byte is written in forward, read in backward, and its gradient
    is written and read once (4 passes). Logits move 12 bytes per entry. The sum is a bound only if no kernel
    overlaps compute with traffic. So it is a model, not a proof.
    """
    from lm.analysis.scaling import flops_per_token

    F = flops_per_token(cfg, T, training=True) * B * T * K
    Q = (4 * activation_bytes(cfg, B, T) + logits_bytes(cfg, B, T)) * K
    return F / P + Q / beta
