"""E1 Task 3: find an induction head and verify it causally.

Without --ckpt, train a small model on random tokens with one repeated segment of random period and start.
Evaluate on s s with |s| = half. Log the phase change during training.
"""
import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from lm.interp.circuits import (composition_null, composition_scores, model_copying_score, prefix_matching_score,
                             previous_token_score)
from lm.interp.common import heatmap, load_model, save_json
from lm.interp.hooks import hooks, run_with_cache
from lm.model import GPT, Config

Head = tuple[int, int]


def repeated(n: int, half: int, vocab: int, g: torch.Generator, device) -> torch.Tensor:
    s = torch.randint(0, vocab, (n, half), generator=g, device=device)
    return torch.cat([s, s], 1)


def mixed_repeats(n: int, T: int, vocab: int, g: torch.Generator, device) -> torch.Tensor:
    """Random tokens with one copied segment: period p in [8, T/2], random start. A fixed-offset head cannot solve it.

    With a constant period (s s only), a head that attends at offset -(half - 1) gets the same loss from position
    alone. This is not induction. Random periods force a content match.
    """
    r = torch.randint(0, vocab, (n, T), generator=g, device=device)
    p = torch.randint(8, T // 2 + 1, (n, 1), generator=g, device=device)
    a = (torch.rand(n, 1, generator=g, device=device) * (T - 2 * p + 1)).long()
    t = torch.arange(T, device=device)
    copy = (t >= a + p) & (t < a + 2 * p)
    return torch.where(copy, r.gather(1, (t - p).clamp(min=0).expand(n, T)), r)


def per_position_loss(model: GPT, x: torch.Tensor) -> torch.Tensor:
    """(T - 1,) mean loss of predicting x[:, i + 1] at position i."""
    logits = model(x)[:, :-1].float()
    return F.cross_entropy(logits.transpose(1, 2), x[:, 1:], reduction="none").mean(0)


def scores(model: GPT, x: torch.Tensor, half: int) -> tuple[torch.Tensor, torch.Tensor]:
    L, H = model.cfg.n_layers, model.cfg.n_heads
    _, cache = run_with_cache(model, x, [f"blocks.{l}.attn.hook_pattern" for l in range(L)])
    pre, prev = torch.zeros(L, H), torch.zeros(L, H)
    for l in range(L):
        p = cache[f"blocks.{l}.attn.hook_pattern"]
        for h in range(H):
            pre[l, h], prev[l, h] = prefix_matching_score(p[:, h], half), previous_token_score(p[:, h])
    return pre, prev


def train(args, device) -> tuple[GPT, list[dict]]:
    torch.manual_seed(args.seed)
    cfg = Config(vocab_size=args.vocab, d_model=args.d_model, n_layers=args.layers, n_heads=args.heads,
                 n_kv_heads=args.heads, max_seq_len=2 * args.half)
    model = GPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    g = torch.Generator(device=device).manual_seed(args.seed)
    x_eval = repeated(64, args.half, args.vocab, torch.Generator(device=device).manual_seed(999), device)
    log, t0 = [], time.perf_counter()
    for step in range(args.steps + 1):
        if step % args.log_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                ppl = per_position_loss(model, x_eval)
            pre, _ = scores(model, x_eval, args.half)
            rec = {"step": step, "tokens": step * args.B * 2 * args.half, "loss_first": ppl[: args.half - 1].mean().item(),
                   "loss_second": ppl[args.half :].mean().item(), "best_prefix": pre.max().item()}
            log.append(rec)
            print(rec, flush=True)
            model.train()
        if step == args.steps:
            break
        for grp in opt.param_groups:
            grp["lr"] = args.lr * min(1, (step + 1) / 100) * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        x = mixed_repeats(args.B, 2 * args.half, args.vocab, g, device)
        with torch.autocast(device, torch.bfloat16, enabled=device == "cuda"):
            loss = F.cross_entropy(model(x)[:, :-1].flatten(0, 1).float(), x[:, 1:].flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"trained in {time.perf_counter() - t0:.0f}s")
    return model.eval(), log


def replace_heads(heads: list[Head], source: dict[str, torch.Tensor]) -> dict:
    """Hooks that set hook_z of the given heads to their values in `source` (a cache)."""
    by_layer: dict[int, list[int]] = {}
    for l, h in heads:
        by_layer.setdefault(l, []).append(h)

    def fn(l, hs):
        def f(o):
            o = o.clone()
            o[:, :, hs] = source[f"blocks.{l}.attn.hook_z"][:, :, hs].to(o.dtype)
            return o
        return f

    return {f"blocks.{l}.attn.hook_z": fn(l, hs) for l, hs in by_layer.items()}


@torch.no_grad()
def second_half_loss(model: GPT, x: torch.Tensor, half: int, fwd: dict | None = None) -> float:
    with hooks(model, fwd or {}):
        return per_position_loss(model, x)[half:].mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="", help="Tier 1 checkpoint; empty = train a small model here")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--vocab", type=int, default=512)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--half", type=int, default=64)
    ap.add_argument("--n_seq", type=int, default=200)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/e1_induction")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log = []
    if args.ckpt:
        model = load_model(args.ckpt, device)
        args.vocab = model.cfg.vocab_size - 64  # stay below the special tokens and the padding rows
    else:
        model, log = train(args, device)
    cfg, half, L, H = model.cfg, args.half, model.cfg.n_layers, model.cfg.n_heads
    g = torch.Generator(device=device).manual_seed(args.seed + 1)
    x = repeated(args.n_seq, half, args.vocab, g, device)
    res: dict = {"train_log": log}

    # 1. Behaviour.
    with torch.no_grad():
        ppl = per_position_loss(model, x).cpu()
    res["loss_first_half"], res["loss_second_half"] = ppl[: half - 1].mean().item(), ppl[half:].mean().item()
    res["loss_uniform"] = math.log(args.vocab)

    # 2. Candidates.
    pre, prev = scores(model, x, half)
    copy = torch.tensor([[model_copying_score(model, l, h) for h in range(H)] for l in range(L)])
    flat = pre.flatten().argsort(descending=True)
    cands = [(int(i) // H, int(i) % H) for i in flat[:2]]
    prev_masked = prev.clone()
    prev_masked[cands[0][0] :] = -1  # a previous-token head must sit in an earlier layer
    p = int(prev_masked.flatten().argmax())
    prev_head = (p // H, p % H)
    # Mean attention at offsets 0..3 and to position 0, for the layer below the top candidate.
    _, pc = run_with_cache(model, x[:32], [f"blocks.{l}.attn.hook_pattern" for l in range(L)])
    pl, i = pc[f"blocks.{max(cands[0][0] - 1, 0)}.attn.hook_pattern"].mean(0), torch.arange(8, 2 * half, device=device)
    res["offset_profile_below"] = [[pl[h, i, i - o].mean().item() for o in range(4)] + [pl[h, i, 0].mean().item()]
                                   for h in range(H)]
    res.update(prefix=pre.tolist(), prev=prev.tolist(), copying=copy.tolist(), candidates=cands, prev_head=prev_head)

    # 3. Mechanism: K-composition from every earlier head into the top candidate, against a null.
    l2, h2 = cands[0]
    kcomp = torch.zeros(L, H)
    for l1 in range(l2):
        for h1 in range(H):
            kcomp[l1, h1] = composition_scores(model, l1, h1, l2, h2)["K"]
    null = composition_null(cfg.d_model, cfg.d_head)
    res.update(kcomp=kcomp.tolist(), kcomp_null_mean=null.mean().item(), kcomp_null_std=null.std().item())

    # 4. Causation. Resample ablation: hook_z from other random sequences of the same shape.
    _, other = run_with_cache(model, repeated(args.n_seq, half, args.vocab, g, device))
    abl = {"none": second_half_loss(model, x, half)}
    abl[f"top1 {cands[0]}"] = second_half_loss(model, x, half, replace_heads(cands[:1], other))
    abl[f"top2 {cands}"] = second_half_loss(model, x, half, replace_heads(cands, other))
    abl[f"prev-token {prev_head}"] = second_half_loss(model, x, half, replace_heads([prev_head], other))
    abl[f"all heads of layer {cands[0][0] - 1}"] = second_half_loss(
        model, x, half, replace_heads([(cands[0][0] - 1, h) for h in range(H)], other))
    res["resample_ablation_second_half_loss"] = abl
    # Denoise: corrupt the first copy. Then restore only the candidate's output from the clean run.
    xc = x.clone()
    xc[:, :half] = torch.randint(0, args.vocab, (args.n_seq, half), generator=g, device=device)
    _, clean = run_with_cache(model, x)
    lc, lx = abl["none"], second_half_loss(model, xc, half)
    lr_ = second_half_loss(model, xc, half, replace_heads(cands[:1], clean))
    res["denoise"] = {"clean": lc, "corrupt": lx, "restored_top1": lr_, "fraction_recovered": (lx - lr_) / (lx - lc)}
    save_json({k: v for k, v in res.items() if k not in ("train_log",)}, f"{args.out}.json")

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax[0, 0].plot(ppl)
    ax[0, 0].axvline(half, color="k", ls=":")
    ax[0, 0].axhline(math.log(args.vocab), color="r", ls="--", label="log V")
    ax[0, 0].set(title="per-position loss, s s", xlabel="position", ylabel="nats")
    ax[0, 0].legend()
    heatmap(ax[0, 1], pre, "prefix-matching score", "head", "layer")
    heatmap(ax[0, 2], prev, "previous-token score", "head", "layer")
    heatmap(ax[1, 0], copy, "copying score", "head", "layer", center=True)
    heatmap(ax[1, 1], kcomp[:l2], f"K-composition into {cands[0]} (null {null.mean():.3f})", "head", "layer")
    if log:
        t = [r["tokens"] for r in log]
        a2 = ax[1, 2].twinx()
        ax[1, 2].plot(t, [r["loss_first"] for r in log], label="loss 1st half")
        ax[1, 2].plot(t, [r["loss_second"] for r in log], label="loss 2nd half")
        a2.plot(t, [r["best_prefix"] for r in log], "k--", label="best prefix score")
        ax[1, 2].set(title="training: phase change?", xlabel="tokens")
        ax[1, 2].legend(loc="center right")
        a2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=110)
    Path(f"{args.out}_train.json").write_text(json.dumps(log))


if __name__ == "__main__":
    main()
