"""A3 Task 8: rank collapse with and without the residual stream."""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from lm.data.tokens import TokenData
from lm.model import GPT, Config, round_up

VARIANTS = {
    "full": dict(),
    "no_residual": dict(use_residual=False),
    "pure_attention": dict(use_residual=False, use_mlp=False),
}


@torch.no_grad()
def collapse_stats(model, x):
    acts = []
    hooks = [b.register_forward_hook(lambda mod, inp, out: acts.append(out.detach().float())) for b in model.blocks]
    model(x)
    for h in hooks:
        h.remove()
    rows = []
    for H in acts:
        centred = H - H.mean(dim=1, keepdim=True)
        rel = (centred.norm(dim=(1, 2)) / H.norm(dim=(1, 2))).mean().item()
        s = torch.linalg.svdvals(H)
        top = (s[:, :10] / s[:, :1]).mean(0).tolist()
        rows.append({"rel_consensus": rel, "sv_top10": [round(v, 4) for v in top]})
    return rows


def run(name, overrides, data, steps=300, B=16, T=256, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cfg = Config(vocab_size=round_up(data.meta["vocab_size"], 64), d_model=256, n_layers=8, n_heads=4,
                 n_kv_heads=2, max_seq_len=T, pos_encoding="nope", **overrides)
    model = GPT(cfg).cuda()
    xv, _ = data.val_windows(8, T)
    out = {"init": collapse_stats(model, xv)}
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
    for _ in range(steps):
        x, y = data.get_batch(B, T, rng=rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(x).flatten(0, 1).float(), y.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    out["trained"] = collapse_stats(model, xv)
    out["final_loss"] = loss.item()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    steps = ap.parse_args().steps
    data = TokenData()
    res = {k: run(k, v, data, steps=steps) for k, v in VARIANTS.items()}
    json.dump(res, open("runs/a3_ablation.json", "w"), indent=1)
    for k, r in res.items():
        print(f"\n{k}  (train loss after {steps} steps {r['final_loss']:.3f})")
        for phase in ("init", "trained"):
            rel = " ".join(f"{row['rel_consensus']:.3f}" for row in r[phase])
            print(f"  {phase:>7} rel. consensus by layer: {rel}")
        print(f"  trained, last layer sigma_2/sigma_1 = {r['trained'][-1]['sv_top10'][1]:.4f}")


if __name__ == "__main__":
    main()
