"""B2 Task 4. ||dW||_2 / ||W||_2 per hidden matrix every 50 steps, AdamW vs Muon. Power iteration, no SVD."""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lm.model.mup import make_model
from lm.train.trainer import train_run
from lm.data.tokens import load_data
from experiments.b2.bakeoff import make_opts

KINDS = ("wq", "wk", "wv", "wo", "w1", "w3", "w2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opts", nargs="+", default=["adamw", "muon"])
    ap.add_argument("--lrs", type=float, nargs="+", default=[3e-3, 0.02])
    ap.add_argument("--adam_lr", type=float, default=3e-3)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--track_every", type=int, default=50)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    data, val = load_data(val_windows=32, T=args.T)
    out = {}
    fig, axs = plt.subplots(1, len(args.opts), figsize=(6 * len(args.opts), 4), sharey=True, squeeze=False)
    for ax, name, lr in zip(axs[0], args.opts, args.lrs):
        model = make_model(args.width, args.layers, "mup", data.meta["vocab_size"], args.T, base_width=args.width, seed=args.seed)
        hist, ups = train_run(model, make_opts(name, model, lr, args.adam_lr, 0.0), data, val, B=args.B, T=args.T,
                          steps=args.steps, warmup=40, eval_every=args.steps, seed=args.seed,
                          track_every=args.track_every)
        # Skip step 0: warm-up makes the first update tiny for every optimiser.
        mats = [n for n in ups[0] if n != "step"]
        R = np.array([[u[n] for n in mats] for u in ups[1:]])          # (probes, matrices)
        out[name] = {"lr": lr, "val": hist[-1]["val_loss"], "steps": [u["step"] for u in ups[1:]], "mats": mats, "ratios": R.tolist()}
        med = np.median(R, axis=0)
        print(f"{name} lr={lr}: final val {hist[-1]['val_loss']:.4f}; median ratio over time per matrix: "
              f"min {med.min():.2e} max {med.max():.2e} max/min {med.max() / med.min():.1f}; "
              f"CV across matrices {med.std() / med.mean():.2f}")
        for k in KINDS:
            idx = [i for i, n in enumerate(mats) if f".{k}." in n]
            layer = [int(mats[i].split(".")[1]) for i in idx]
            ax.plot(layer, med[idx], "o-", label=k)
        ax.set(title=f"{name} lr={lr}", xlabel="layer", yscale="log", xticks=range(args.layers))
        ax.legend(ncol=2, fontsize=7)
    axs[0, 0].set_ylabel(r"median over steps of $\|\Delta W\|_2 / \|W\|_2$")
    fig.suptitle(f"Update size per hidden matrix: d={args.width}, L={args.layers}, B={args.B}, T={args.T}, {args.steps} steps, "
                 f"probe every {args.track_every}, seed {args.seed}")
    fig.tight_layout()
    fig.savefig("runs/b2_update_ratios.png", dpi=120)
    json.dump({"args": vars(args), **out}, open("runs/b2_update_ratios.json", "w"))


if __name__ == "__main__":
    main()
