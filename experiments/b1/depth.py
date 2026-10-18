"""B1 Task 5. Stream RMS after each sublayer at initialisation, with and without 1/sqrt(2L) on the output projections."""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lm.model.mup import make_model
from lm.data.tokens import load_data


@torch.no_grad()
def stream_rms2(model, x) -> np.ndarray:
    """||h_k||_rms^2 for k = 0 (embedding) .. 2L. Hooks on the norms read the stream before each sublayer."""
    out = []
    norms = [m for b in model.blocks for m in (b.attn_norm, b.mlp_norm)]
    hooks = [n.register_forward_pre_hook(lambda m, a: out.append(a[0].float().pow(2).mean().item())) for n in norms]
    hooks.append(model.norm.register_forward_pre_hook(lambda m, a: out.append(a[0].float().pow(2).mean().item())))
    model(x)
    for h in hooks:
        h.remove()
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depths", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--T", type=int, default=256)
    args = ap.parse_args()
    data, (x, _) = load_data(val_windows=8, T=args.T)
    res = {}
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, scaled in zip(axs, (False, True)):
        for L in args.depths:
            r = np.mean([stream_rms2(make_model(args.width, L, "sp", data.meta["vocab_size"], args.T,
                                                base_width=args.width, seed=s, scale_out_proj=scaled), x)
                         for s in args.seeds], axis=0)
            res[f"L={L} scaled={scaled}"] = r.tolist()
            inc = np.polyfit(np.arange(len(r)), r, 1)[0]
            print(f"L={L:2d} scale_out_proj={scaled!s:5}: rms2 h0={r[0]:.2e} h_2L={r[-1]:.2e} "
                  f"slope/sublayer={inc:.2e} added/embedding={(r[-1] - r[0]) / r[0]:.2f}")
            ax.plot(np.arange(len(r)), r, "o-", ms=2, label=f"L={L}")
        ax.set(title=f"scale_out_proj={scaled}", xlabel="sublayer k")
        ax.legend()
    axs[0].set_ylabel(r"$\|h_k\|_{rms}^2$ at init")
    fig.suptitle(f"Stream at init: width {args.width}, init_std 0.02, seeds {args.seeds}, T={args.T}, real data")
    fig.tight_layout()
    fig.savefig("runs/b1_depth.png", dpi=120)
    json.dump({"args": vars(args), "rms2": res}, open("runs/b1_depth.json", "w"))


if __name__ == "__main__":
    main()
