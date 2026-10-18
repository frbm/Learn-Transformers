"""D4 speculative decoding: the exact accept/resample rule, the toy algorithm, and the KV-cached version."""
import torch
from torch import Tensor

from lm.infer.generate import BatchCache, forward_cached
from lm.infer.sampling import Sampler


def accept_or_resample(p: Tensor, q: Tensor, x: int, generator=None) -> tuple[bool, int]:
    """Proposition 5. Accept x ~ q with prob min(1, p/q). Else draw from (p - q)+ / ||(p - q)+||."""
    u = torch.rand((), generator=generator, device=p.device, dtype=p.dtype)
    # u * q < p avoids the division. If q[x] <= p[x], the test always passes.
    if bool(u * q[x] < p[x]):
        return True, x
    r = (p - q).clamp(min=0)
    return False, int(torch.multinomial(r / r.sum(), 1, generator=generator))


def speculative_generate(p_fn, q_fn, prefix: list[int], n_tokens: int, gamma: int,
                         generator=None) -> tuple[list[int], dict]:
    seq = list(prefix)
    stats = {"rounds": 0, "proposed": 0, "accepted": 0, "target_calls": 0}
    while (left := n_tokens - (len(seq) - len(prefix))) > 0:
        g = min(gamma, left - 1)                             # never propose past n_tokens: the round gives <= g + 1
        drafts, qs = [], []
        for _ in range(g):
            q = q_fn([seq + drafts])[0]
            drafts.append(int(torch.multinomial(q, 1, generator=generator)))
            qs.append(q)
        ps = p_fn([seq + drafts[:i] for i in range(g + 1)])  # one target call on the g + 1 prefixes
        stats["rounds"] += 1; stats["proposed"] += g; stats["target_calls"] += 1
        for i in range(g):
            ok, y = accept_or_resample(ps[i], qs[i], drafts[i], generator)
            seq.append(y)
            if not ok:
                break
            stats["accepted"] += 1
        else:                                                # all accepted: one free token from the last target row
            seq.append(int(torch.multinomial(ps[g], 1, generator=generator)))
    return seq[len(prefix):], stats


def expected_tokens_per_call(alpha: float, gamma: int) -> float:
    """Proposition 6: E[N] = (1 - alpha^(gamma+1)) / (1 - alpha)."""
    return float(gamma + 1) if alpha == 1.0 else (1 - alpha ** (gamma + 1)) / (1 - alpha)


def expected_speedup(alpha: float, gamma: int, c: float) -> float:
    return expected_tokens_per_call(alpha, gamma) / (1 + gamma * c)


@torch.no_grad()
def speculative_cached(target, draft, prompt: list[int], n_tokens: int, gamma: int, sampler: Sampler,
                       generator=None) -> tuple[list[int], dict]:
    """Speculative sampling on two GPTs with KV caches, batch 1.

    Invariant at the start of a round: the target cache holds seq[:-1]. The draft cache lags by 1 or more tokens
    and catches up in one short prefill. A rejection rolls both caches back. Nothing is copied.
    The sampler's processors apply to both models before the comparison, so the output law is the processed p.
    """
    dev = target.tok_emb.weight.device
    n_slots = len(prompt) + n_tokens + gamma + 2
    tc = BatchCache(target.cfg, 1, n_slots, dev, target.tok_emb.weight.dtype)
    dc = BatchCache(draft.cfg, 1, n_slots, dev, draft.tok_emb.weight.dtype)
    dist = lambda logits: sampler.process(logits.float()).softmax(-1).double()
    run = lambda model, cache, toks: forward_cached(model, torch.tensor([toks], device=dev), cache)[0]
    seq = list(prompt)
    if len(seq) > 1:
        run(target, tc, seq[:-1])
    reached, acc = torch.zeros(gamma, dtype=torch.long), torch.zeros(gamma, dtype=torch.long)
    tv_sum = torch.zeros(gamma, dtype=torch.float64)
    stats = {"rounds": 0, "proposed": 0, "accepted": 0, "target_calls": 0}
    while (left := n_tokens - (len(seq) - len(prompt))) > 0:
        g, L = min(gamma, left - 1), len(seq)
        drafts, qs = [], []
        if g > 0:
            q = dist(run(draft, dc, seq[dc.len:])[-1])
            for j in range(g):
                d = int(torch.multinomial(q, 1, generator=generator))
                drafts.append(d)
                qs.append(q)
                if j < g - 1:
                    q = dist(run(draft, dc, [d])[-1])
        ps = dist(run(target, tc, [seq[-1]] + drafts))       # (g + 1, V) in one pass
        stats["rounds"] += 1; stats["proposed"] += g; stats["target_calls"] += 1
        n_acc = g
        for i in range(g):
            reached[i] += 1
            tv_sum[i] += 0.5 * (ps[i] - qs[i]).abs().sum().item()
            ok, y = accept_or_resample(ps[i], qs[i], drafts[i], generator)
            seq.append(y)
            if not ok:
                n_acc = i
                break
            acc[i] += 1
        else:
            seq.append(int(torch.multinomial(ps[g], 1, generator=generator)))
        stats["accepted"] += n_acc
        tc.rollback(L + n_acc)                               # keep seq[:-1] only: the last token is not yet fed
        dc.rollback(min(dc.len, L + n_acc))
    stats["accept_rate_at"] = (acc / reached.clamp(min=1)).tolist()
    stats["one_minus_tv_at"] = (1 - tv_sum / reached.clamp(min=1)).tolist()
    stats["reached_at"] = reached.tolist()
    return seq[len(prompt):], stats
