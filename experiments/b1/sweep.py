"""B1 Task 4. Learning-rate transfer sweep: width x lr x {SP, μP}. Resumable: one JSON line per finished run."""
import argparse
import json
import math
import time
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


def key(r: dict) -> tuple:
    return r["param"], r["width"], r["log2_lr"], r["seed"]


def done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    # A run killed mid-write leaves a partial last line. Skip it; the run repeats.
    out = set()
    for line in path.read_text().splitlines():
        try:
            out.add(key(json.loads(line)))
        except (json.JSONDecodeError, KeyError):
            pass
    return out


def sweep(args) -> None:
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = done_keys(path)
    data, val = load_data(val_windows=args.val_windows, T=args.T)
    todo = [(p, w, e, s) for s in args.seeds for p in args.params for w in args.widths for e in args.log2_lrs
            if (p, w, e, s) not in done]
    print(f"{len(done)} runs done, {len(todo)} to do -> {path}")
    for i, (par, width, e, seed) in enumerate(todo):
        model = make_model(width, args.layers, par, data.meta["vocab_size"], args.T, base_width=args.base_width, seed=seed)
        opt = torch.optim.AdamW(param_groups(model, 2.0**e, args.wd), betas=(0.9, 0.95), fused=True)
        t0 = time.perf_counter()
        hist, _ = train_run(model, [opt], data, val, B=args.B, T=args.T, steps=args.steps, warmup=args.warmup,
                        eval_every=args.steps, seed=seed)
        rec = {"param": par, "width": width, "log2_lr": e, "seed": seed, "steps": args.steps, "B": args.B, "T": args.T,
               "layers": args.layers, "base_width": args.base_width, "wd": args.wd, "val_loss": hist[-1]["val_loss"],
               "diverged": not math.isfinite(hist[-1]["val_loss"]), "wall_s": time.perf_counter() - t0}
        with path.open("a") as f:                                    # append and close: survives a kill
            f.write(json.dumps(rec) + "\n")
        print(f"[{i + 1}/{len(todo)}] {par} w={width} lr=2^{e} val={rec['val_loss']:.4f} ({rec['wall_s']:.0f}s)", flush=True)
        del model, opt
        torch.cuda.empty_cache()


def plot(path: Path, png: str) -> None:
    rs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    finite = [r["val_loss"] for r in rs if math.isfinite(r["val_loss"])]
    for ax, par in zip(axs, ("sp", "mup")):
        for w in sorted({r["width"] for r in rs if r["param"] == par}):
            pts = sorted((r["log2_lr"], r["val_loss"]) for r in rs if r["param"] == par and r["width"] == w)
            # Mean over seeds per lr.
            lrs = sorted({e for e, _ in pts})
            ys = np.array([np.mean([v for e2, v in pts if e2 == e]) for e in lrs])
            ys = np.where(np.isfinite(ys), ys, np.nan)
            ax.plot(lrs, ys, "o-", label=f"width {w}")
            if np.isfinite(ys).any():
                j = int(np.nanargmin(ys))
                ax.plot(lrs[j], ys[j], "k*", ms=10)
                print(f"{par} width {w}: optimum lr = 2^{lrs[j]}  val {ys[j]:.4f}")
        ax.set(title=par.upper(), xlabel=r"$\log_2 \eta$")
        if finite:
            ax.set_ylim(min(finite) - 0.05, min(finite) + 1.5)
        ax.legend()
    axs[0].set_ylabel("final val loss (nats)")
    r0 = rs[0]
    fig.suptitle(f"LR transfer: {r0['steps']} steps, L={r0['layers']}, base={r0['base_width']}, B={r0['B']}, T={r0['T']}, "
                 f"wd={r0['wd']}, seeds={sorted({r['seed'] for r in rs})}")
    fig.tight_layout()
    fig.savefig(png, dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", nargs="+", default=["sp", "mup"])
    ap.add_argument("--widths", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--log2_lrs", type=int, nargs="+", default=list(range(-14, -5)))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--base_width", type=int, default=128)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--val_windows", type=int, default=64)
    ap.add_argument("--out", default="runs/b1_sweep.jsonl")
    ap.add_argument("--plot_only", action="store_true")
    args = ap.parse_args()
    if not args.plot_only:
        sweep(args)
    plot(Path(args.out), args.out.replace(".jsonl", ".png"))


if __name__ == "__main__":
    main()
