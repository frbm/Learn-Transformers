"""B2 Task 5. AdamW vs Muon (+AdamW) vs SGD-momentum on Tier 0 FineWeb: lr grid, then 3 seeds at a fixed wall-clock budget.

Resumable: one JSON line per finished run in --out, keyed by (stage, opt, lr, seed, budget).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lm.model.mup import make_model
from lm.train.trainer import train_run
from lm.data.tokens import load_data
from lm.model.mup import param_groups
from lm.train.optim import build_optimizers

GRIDS = {"adamw": [1e-3, 3e-3, 1e-2], "muon": [0.01, 0.02, 0.04], "sgd": [0.1, 0.3, 1.0]}


def make_opts(name: str, model, lr: float, adam_lr: float, wd: float) -> list[torch.optim.Optimizer]:
    if name == "adamw":
        return [torch.optim.AdamW(param_groups(model, lr, wd), betas=(0.9, 0.95), fused=True)]
    if name == "muon":
        return build_optimizers(model, muon_lr=lr, adam_lr=adam_lr, weight_decay=wd)
    if name == "sgd":
        return [torch.optim.SGD(param_groups(model, lr, wd), lr=lr, momentum=0.9, nesterov=True)]
    raise ValueError(name)


def run(stage: str, name: str, lr: float, seed: int, budget: float, data, val, args) -> dict:
    model = make_model(args.width, args.layers, "mup", data.meta["vocab_size"], args.T, base_width=args.width, seed=seed)
    opts = make_opts(name, model, lr, args.adam_lr, args.wd)
    hist, _ = train_run(model, opts, data, val, B=args.B, T=args.T, budget_s=budget, warmup=args.warmup,
                    eval_every=args.eval_every, seed=seed)
    return {"stage": stage, "opt": name, "lr": lr, "seed": seed, "budget": budget, "adam_lr": args.adam_lr, "wd": args.wd,
            "width": args.width, "layers": args.layers, "B": args.B, "T": args.T, "final": hist[-1]["val_loss"],
            "steps": hist[-1]["step"], "hist": [(h["time_s"], h["val_loss"]) for h in hist]}


def time_to(hist: list, target: float) -> float:
    """First time the val loss reaches target, linear interpolation between evaluations. inf if never."""
    for (t0, l0), (t1, l1) in zip(hist, hist[1:]):
        if l1 <= target:
            return t1 if l0 == l1 else t0 + (t1 - t0) * (l0 - target) / (l0 - l1)
    return float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opts", nargs="+", default=["adamw", "muon", "sgd"])
    ap.add_argument("--tune_budget", type=float, default=30.0)
    ap.add_argument("--budget", type=float, default=90.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--adam_lr", type=float, default=3e-3, help="AdamW lr for Muon's non-hidden tensors")
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--val_windows", type=int, default=64)
    ap.add_argument("--out", default="runs/b2_bakeoff.jsonl")
    args = ap.parse_args()

    path = Path(args.out)
    rs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
    have = {(r["stage"], r["opt"], r["lr"], r["seed"], r["budget"]) for r in rs}
    data, val = load_data(val_windows=args.val_windows, T=args.T)

    def get(stage, name, lr, seed, budget):
        k = (stage, name, lr, seed, budget)
        if k not in have:
            r = run(*k, data, val, args)
            with path.open("a") as f:
                f.write(json.dumps(r) + "\n")
            rs.append(r)
            have.add(k)
            print(f"{stage} {name} lr={lr} seed={seed}: val {r['final']:.4f} after {r['steps']} steps", flush=True)
        return next(r for r in rs if (r["stage"], r["opt"], r["lr"], r["seed"], r["budget"]) == k)

    # Stage 1: tune each optimiser on its grid. Same seed, same short budget.
    best = {}
    for name in args.opts:
        tuned = [get("tune", name, lr, 0, args.tune_budget) for lr in GRIDS[name]]
        best[name] = min(tuned, key=lambda r: r["final"])["lr"]
        if best[name] in (GRIDS[name][0], GRIDS[name][-1]):
            print(f"warning: {name} optimum {best[name]} is on the edge of the grid {GRIDS[name]}")
    print("best lr:", best)

    # Stage 2: best lr, all seeds, the full wall-clock budget.
    final = {name: [get("final", name, best[name], s, args.budget) for s in args.seeds] for name in args.opts}
    ref = "adamw" if "adamw" in final else args.opts[0]
    target = float(np.mean([r["final"] for r in final[ref]]))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for c, (name, runs) in zip(plt.cm.tab10.colors, final.items()):
        fin = np.array([r["final"] for r in runs])
        tt = np.array([time_to(r["hist"], target) for r in runs])
        steps = np.mean([r["steps"] for r in runs])
        ok = tt[np.isfinite(tt)]
        reach = f"{ok.mean():.1f} s (range {ok.min():.1f}-{ok.max():.1f})" if len(ok) else "never"
        print(f"{name:6s} lr={best[name]:<6g} final val {fin.mean():.4f} +- {fin.std():.4f} (min {fin.min():.4f}, max {fin.max():.4f}); "
              f"time to {target:.4f}: {reach}, reached by {len(ok)}/{len(tt)} seeds; {steps:.0f} steps")
        for i, r in enumerate(runs):
            t, l = zip(*r["hist"][1:])
            ax.plot(t, l, color=c, alpha=0.7, label=f"{name} lr={best[name]}" if i == 0 else None)
    ax.axhline(target, color="k", ls=":", lw=0.8)
    ax.set(xlabel="training wall time (s, eval excluded)", ylabel="val loss (nats)", ylim=(target - 0.3, target + 1.5),
           title=f"Bake-off: d={args.width}, L={args.layers}, B={args.B}, T={args.T}, budget {args.budget:g}s, "
                 f"seeds {args.seeds}, {torch.cuda.get_device_name()}")
    ax.title.set_fontsize(8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=120)


if __name__ == "__main__":
    main()
