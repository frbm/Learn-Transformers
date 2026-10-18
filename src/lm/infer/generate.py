"""D4 batched generation engine: left padding, per-row RoPE positions, a rollback-able KV cache, per-row stop."""
import torch
from torch import Tensor

from lm.infer.sampling import Sampler, truncation_kl
from lm.model.cache import BatchCache


def forward_cached(model, idx: Tensor, cache: BatchCache, valid_new: Tensor | None = None,
                   slot: Tensor | None = None, last_only: bool = False) -> Tensor:
    """Logits (B, T, V) for idx written at slots [len, len + T). last_only: logits (B, 1, V) of the last slot.

    With slot (a 1-element device tensor) the shapes are static: attend over all slots, do not move `len`.
    """
    return model(idx, cache, valid=valid_new, slot=slot, last_only=last_only)


def left_pad(prompts: list[list[int]], pad_id: int, device) -> tuple[Tensor, Tensor]:
    L = max(map(len, prompts))
    idx = torch.full((len(prompts), L), pad_id, dtype=torch.long)
    valid = torch.zeros(len(prompts), L, dtype=torch.bool)
    for i, p in enumerate(prompts):
        if p:
            idx[i, L - len(p):], valid[i, L - len(p):] = torch.tensor(p), True
    return idx.to(device), valid.to(device)


def _cut(rows: list[list[int]], eos_id: int) -> list[list[int]]:
    return [r[: r.index(eos_id)] if eos_id in r else r for r in rows]


@torch.no_grad()
def generate(model, prompts: list[list[int]], max_new: int, sampler: Sampler, eos_id: int,
             generator=None, return_logq: bool = False, pad_id: int = 0, constraint=None):
    """New tokens per prompt, cut before the first eos.

    return_logq=True also returns log q(y) under the PROCESSED distributions, eos term included (F1 Proposition 1).
    constraint: an object with apply(logits) -> logits (token masking, before the sampler) and step(tokens).
    """
    dev, dtype = model.tok_emb.weight.device, model.tok_emb.weight.dtype
    B = len(prompts)
    idx, valid = left_pad(prompts, pad_id, dev)
    cache = BatchCache(model.cfg, B, idx.shape[1] + max_new, dev, dtype)
    logits = forward_cached(model, idx, cache, valid, last_only=True)[:, -1]
    out = torch.full((B, max_new), eos_id, dtype=torch.long, device=dev)
    logq = torch.zeros(B, dtype=torch.float64, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    for t in range(max_new):
        logits = logits.float()
        if constraint is not None:
            logits = constraint.apply(logits)
        proc = sampler.process(logits)
        nxt = sampler.draw(proc, generator).masked_fill(done, eos_id)
        if constraint is not None:
            constraint.step(nxt)
        logq += proc.log_softmax(-1).gather(-1, nxt[:, None])[:, 0].masked_fill(done, 0.0).double()
        out[:, t] = nxt
        done |= nxt == eos_id
        # done.all() syncs with the GPU. Check it only every 16 steps.
        if t == max_new - 1 or (t % 16 == 15 and bool(done.all())):
            break
        logits = forward_cached(model, nxt[:, None], cache)[:, -1]
    seqs = _cut(out.tolist(), eos_id)
    return (seqs, logq.tolist()) if return_logq else seqs


@torch.no_grad()
def rescore(model, prompts: list[list[int]], responses: list[list[int]], processor=None, pad_id: int = 0,
            batch_size: int = 64, per_token: bool = False) -> dict:
    """Per-row sums over response tokens, in padded forward passes (float64 sums):

    logp: log p under the raw model. logq: log q under softmax(processor(logits)).
    trunc: sum_t -log p_t(S_t), the truncation price of Proposition 3. n: the number of response tokens.
    per_token: also "tok_logp", a list of (n_i,) tensors of raw log p per response token.
    """
    dev, dtype = model.tok_emb.weight.device, model.tok_emb.weight.dtype
    out = {k: [] for k in ("logp", "logq", "trunc", "n")}
    toks: list[Tensor] = []
    for b in range(0, len(prompts), batch_size):
        P, R = prompts[b : b + batch_size], responses[b : b + batch_size]
        idx, valid = left_pad([p + r for p, r in zip(P, R)], pad_id, dev)
        cache = BatchCache(model.cfg, len(P), idx.shape[1], dev, dtype)
        logits = forward_cached(model, idx, cache, valid)[:, :-1].float()
        L = idx.shape[1]
        n = torch.tensor([len(r) for r in R], device=dev)
        is_resp = torch.arange(1, L, device=dev)[None] >= (L - n)[:, None]
        tgt = idx[:, 1:, None]
        logp = logits.log_softmax(-1)
        lp_tok = logp.gather(-1, tgt)[..., 0].masked_fill(~is_resp, 0).double()
        out["logp"].append(lp_tok.sum(1))
        if per_token:
            toks += [row[L - 1 - k :].cpu() if k else row[:0].cpu() for row, k in zip(lp_tok, n.tolist())]
        if processor is None:
            out["logq"].append(out["logp"][-1]); out["trunc"].append(torch.zeros_like(out["logp"][-1]))
        else:
            proc = processor(logits)
            out["logq"].append(proc.log_softmax(-1).gather(-1, tgt)[..., 0].masked_fill(~is_resp, 0).double().sum(1))
            out["trunc"].append(truncation_kl(logits, proc).masked_fill(~is_resp, 0).double().sum(1))
        out["n"].append(n)
    res = {k: torch.cat(v).cpu() for k, v in out.items()}
    if per_token:
        res["tok_logp"] = toks
    return res


def sequence_logprob(model, prompts: list[list[int]], responses: list[list[int]], pad_id: int = 0,
                     processor=None) -> Tensor:
    """sum_t log q_t(y_t | prompt, y_<t) per row. q_t = softmax(processor(logits))."""
    return rescore(model, prompts, responses, processor, pad_id)["logq"]


@torch.no_grad()
def beam_search(model, prompt: list[int], k: int, max_new: int, eos_id: int, length_alpha: float = 0.0) -> list[int]:
    """Best finished hypothesis by sum log p / len ** length_alpha. Approximates the mode."""
    dev, dtype = model.tok_emb.weight.device, model.tok_emb.weight.dtype
    cache = BatchCache(model.cfg, k, len(prompt) + max_new, dev, dtype)
    logits = forward_cached(model, torch.tensor([prompt] * k, device=dev), cache)[:, -1]
    scores = torch.full((k,), float("-inf"), device=dev)
    scores[0] = 0.0                                          # k copies of one prompt: start from one beam
    hyps = torch.zeros(k, 0, dtype=torch.long, device=dev)
    finished: list[tuple[float, list[int]]] = []
    norm = lambda s, n: s / max(n, 1) ** length_alpha
    for t in range(max_new):
        cand = (scores[:, None] + logits.float().log_softmax(-1)).flatten()
        top, flat = cand.topk(2 * k)
        rows, toks = flat // logits.shape[-1], flat % logits.shape[-1]
        is_eos = toks == eos_id
        for s, r in zip(top[is_eos].tolist(), rows[is_eos].tolist()):
            finished.append((norm(s, t + 1), hyps[r].tolist()))
        keep = (~is_eos).nonzero()[:k, 0]
        scores, rows, toks = top[keep], rows[keep], toks[keep]
        if len(finished) >= k and max(f[0] for f in finished) >= norm(scores[0].item(), t + 1):
            break
        hyps = torch.cat([hyps[rows], toks[:, None]], 1)
        cache.reorder(rows)
        logits = forward_cached(model, toks[:, None], cache)[:, -1]
    if not finished:
        return hyps[0].tolist()
    return max(finished, key=lambda f: f[0])[1]


def kv_cache_bytes(n_layers: int, n_kv_heads: int, head_dim: int, n_tokens: int, bytes_per_value: int = 2) -> int:
    return 2 * n_layers * n_kv_heads * head_dim * n_tokens * bytes_per_value


def decode_bound(B: int, W: float, N: float, K: float, P: float, beta: float) -> float:
    """Proposition 7: lower bound (s) of one decode step. W, K in bytes, P in FLOP/s, beta in B/s."""
    return max((W + B * K) / beta, 2 * N * B / P)


def _graph_step(model, cache: BatchCache, B: int, start: int):
    """Capture one static-shape decode step as a CUDA graph. Call before the prefill: capture resets the cache."""
    dev = cache.k.device
    tok_buf = torch.zeros(B, 1, dtype=torch.long, device=dev)
    slot = torch.tensor([start], device=dev)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):                                   # warm up kernels and the allocator off the graph
            forward_cached(model, tok_buf, cache, slot=slot)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = forward_cached(model, tok_buf, cache, slot=slot)[:, -1].argmax(-1, keepdim=True)
    cache.n_real.zero_()
    cache.valid.zero_()

    def step(tok: Tensor) -> Tensor:
        tok_buf.copy_(tok)
        graph.replay()
        slot.add_(1)
        return out
    return step


@torch.no_grad()
def time_generation(model, B: int, prompt_len: int, new_tokens: int, cuda_graph: bool = False,
                    seed: int = 0) -> dict[str, float]:
    """TTFT (prefill), ITL (mean decode step) in ms, and decode tokens/s. Greedy argmax, random prompts."""
    dev, dtype = model.tok_emb.weight.device, model.tok_emb.weight.dtype
    g = torch.Generator(device=dev).manual_seed(seed)
    prompt = torch.randint(0, model.cfg.vocab_size, (B, prompt_len), device=dev, generator=g)

    def once() -> tuple[float, float]:
        cache = BatchCache(model.cfg, B, prompt_len + new_tokens + 1, dev, dtype)
        if cuda_graph:
            step = _graph_step(model, cache, B, prompt_len)
        else:
            step = lambda tok: forward_cached(model, tok, cache)[:, -1].argmax(-1, keepdim=True)
        e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        torch.cuda.synchronize()
        e0.record()
        tok = forward_cached(model, prompt, cache, last_only=True)[:, -1].argmax(-1, keepdim=True)
        e1.record()
        for _ in range(new_tokens):
            tok = step(tok)
        e2.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1), e1.elapsed_time(e2)

    once()                                                   # warm-up run: kernel selection, allocator
    ttft, dec = once()
    itl = dec / new_tokens
    return {"ttft_ms": ttft, "itl_ms": itl, "tok_per_s": B * 1000 / itl}
