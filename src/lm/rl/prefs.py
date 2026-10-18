"""D3 §2-4. Preference losses, KL estimators, group advantages, the clipped policy loss."""
import math

import torch
import torch.nn.functional as F
from torch import Tensor


def pad_batch(prompts: list[list[int]], completions: list[list[int]], pad_id: int = 0):
    """Right-pad prompt + completion. Causal attention never looks right, so padding needs no mask."""
    lp = torch.tensor([len(p) for p in prompts])
    lc = torch.tensor([len(c) for c in completions])
    T, Tc = int((lp + lc).max()), max(int(lc.max()), 1)
    idx = torch.full((len(prompts), T), pad_id, dtype=torch.long)
    tgt = torch.full((len(prompts), Tc), pad_id, dtype=torch.long)
    for i, (p, c) in enumerate(zip(prompts, completions)):
        idx[i, : len(p) + len(c)] = torch.tensor(p + c)
        tgt[i, : len(c)] = torch.tensor(c, dtype=torch.long)
    mask = torch.arange(Tc)[None] < lc[:, None]
    # Completion token j is predicted at position |prompt| + j - 1. Clamp the padded slots to a valid index.
    pos = (lp[:, None] - 1 + torch.arange(Tc)[None]).clamp(max=T - 1)
    return idx, tgt, mask, pos


def completion_logits(model, prompts: list[list[int]], completions: list[list[int]]) -> tuple[Tensor, Tensor, Tensor]:
    """fp32 logits at the completion positions (N, Tc, V), targets (N, Tc), mask (N, Tc)."""
    dev = model.tok_emb.weight.device
    idx, tgt, mask, pos = (t.to(dev) for t in pad_batch(prompts, completions))
    h = model.hidden(idx)
    # Gather the hidden states first. Then the readout runs only on completion positions, not on the prompt.
    h = h.gather(1, pos[..., None].expand(-1, -1, h.shape[-1]))
    return readout(model, h), tgt, mask


def readout(model, h: Tensor) -> Tensor:
    """Final norm and lm_head in at least fp32, also under bf16 autocast.

    bf16 logits of size 10 carry errors near 0.05. Then the importance ratios of two code paths
    (cached sampler, full forward) differ by that much. An fp32 readout removes this source.
    """
    dt = torch.promote_types(h.dtype, torch.float32)
    with torch.autocast(h.device.type, enabled=False):
        return F.linear(model.norm(h.to(dt)), model.lm_head.weight.to(dt))


def completion_logprobs(model, prompts: list[list[int]], completions: list[list[int]],
                        per_token: bool = False) -> Tensor | tuple[Tensor, Tensor]:
    z, tgt, mask = completion_logits(model, prompts, completions)
    # log pi(y_t) = z_y - logsumexp(z): no (N, T, V) log-softmax tensor. fp32 for the reduction.
    lp = (z.gather(-1, tgt[..., None])[..., 0] - torch.logsumexp(z, -1)) * mask
    return (lp, mask.to(lp.dtype)) if per_token else lp.sum(-1)


def _reduce(x: Tensor, reduction: str) -> Tensor:
    return {"mean": x.mean, "sum": x.sum, "none": lambda: x}[reduction]()


def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta: float, reduction: str = "mean") -> tuple[Tensor, dict]:
    chosen, rejected = beta * (pi_w - ref_w), beta * (pi_l - ref_l)
    h = chosen - rejected
    # -log sigmoid(h) = softplus(-h): stable for large |h|.
    loss = F.softplus(-h)
    with torch.no_grad():
        m = {"margin": h, "accuracy": (h > 0).to(h.dtype), "chosen_reward": chosen, "rejected_reward": rejected}
        m = {k: _reduce(v.detach(), reduction) for k, v in m.items()}
    return _reduce(loss, reduction), m


def ipo_loss(pi_w, pi_l, ref_w, ref_l, beta: float, reduction: str = "mean") -> Tensor:
    """Azar et al.: squared loss towards the finite target margin 1 / (2 beta). h has no beta factor."""
    h = (pi_w - ref_w) - (pi_l - ref_l)
    return _reduce((h - 1 / (2 * beta)) ** 2, reduction)


def kl_estimators(logp: Tensor, logp_ref: Tensor) -> dict[str, Tensor]:
    """Samples from the policy. u = log(pi_ref / pi). k1 = -u, k2 = u^2 / 2, k3 = e^u - 1 - u."""
    u = logp_ref - logp
    # expm1 keeps k3 accurate when u is near 0, where e^u - 1 cancels.
    return {"k1": -u, "k2": 0.5 * u ** 2, "k3": torch.expm1(u) - u}


def group_advantages(rewards: Tensor, scale: str = "std", eps: float = 1e-6) -> Tensor:
    A = rewards - rewards.mean(-1, keepdim=True)
    if scale == "std":
        A = A / (rewards.std(-1, keepdim=True) + eps)
    elif scale != "none":
        raise ValueError(scale)
    return A


def aggregate(x: Tensor, mask: Tensor, aggregation: str, max_len: int | None = None) -> Tensor:
    if aggregation == "sequence":
        return ((x * mask).sum(-1) / mask.sum(-1).clamp_min(1)).mean()
    if aggregation == "token":
        return (x * mask).sum() / mask.sum().clamp_min(1)
    if aggregation == "constant":
        return (x * mask).sum() / (mask.shape[0] * (max_len or mask.shape[1]))
    raise ValueError(aggregation)


def policy_loss(logp, logp_old, adv, mask, eps_low=0.2, eps_high=0.2, aggregation="sequence",
                logp_ref=None, beta=0.0, max_len=None) -> Tensor:
    mask = mask.to(logp.dtype)
    ratio = torch.exp(logp - logp_old)
    A = adv[:, None]
    # Pessimistic bound: the min removes the incentive to push a ratio outside [1 - eps_low, 1 + eps_high].
    obj = torch.minimum(ratio * A, ratio.clamp(1 - eps_low, 1 + eps_high) * A)
    loss = aggregate(-obj, mask, aggregation, max_len)
    if beta > 0:
        loss = loss + beta * aggregate(kl_estimators(logp, logp_ref)["k3"], mask, aggregation, max_len)
    return loss


def bon_kl(n: int) -> float:
    return math.log(n) - (n - 1) / n
