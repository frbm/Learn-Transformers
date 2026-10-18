"""B1 Task 3. Coordinate check: ||h_t - h_0||_rms per layer after each Adam step, against width."""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from lm.model.mup import make_model
from lm.data.tokens import load_data
from lm.model.mup import param_groups


@torch.no_grad()
def activations(model, x) -> list[torch.Tensor]:
    """Embedding output, every block output, logits. Forward hooks, so the model code stays unchanged."""
    acts = []
    hooks = [model.tok_emb.register_forward_hook(lambda m, i, o: acts.append(o.float()))]
    hooks += [b.register_forward_hook(lambda m, i, o: acts.append(o.float())) for b in model.blocks]
    acts.append(model(x).float())
    for h in hooks:
        h.remove()
    return acts


def coord_check(width: int, par: str, data, probe, args) -> np.ndarray:
    """Returns (steps, n_layers + 2): RMS change of each activation after each step."""
    model = make_model(width, args.layers, par, data.meta["vocab_size"], args.T, base_width=args.base_width, seed=args.seed)
    opt = torch.optim.AdamW(param_groups(model, args.lr), betas=(0.9, 0.95), weight_decay=0.0)
    rng = np.random.default_rng(args.seed)                          # same batches at every width
    h0 = activations(model, probe)
    out = np.zeros((args.steps, len(h0)))
    for t in range(args.steps):
        x, y = data.get_batch(args.B, args.T, rng=rng)
        F.cross_entropy(model(x).flatten(0, 1).float(), y.flatten()).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        out[t] = [(a - b).pow(2).mean().sqrt().item() for a, b in zip(activations(model, probe), h0)]
    return out


def plot(res: dict, widths: list[int], args) -> None:
    names = ["embedding"] + [f"block {i}" for i in range(args.layers)] + ["logits"]
    for par, r in res.items():
        r = np.array([r[str(w)] for w in widths])                  # (widths, steps, layers)
        fig, axs = plt.subplots(1, len(names), figsize=(3 * len(names), 3), sharex=True)
        for j, (ax, nm) in enumerate(zip(axs, names)):
            for t in range(r.shape[1]):
                ax.plot(widths, r[:, t, j], "o-", ms=3, color=plt.cm.viridis(t / max(1, r.shape[1] - 1)), label=f"t={t + 1}")
            ax.set(xscale="log", yscale="log", title=nm, xlabel="width")
        axs[0].set_ylabel(r"$\|h_t - h_0\|_{rms}$")
        axs[-1].legend(fontsize=6)
        fig.suptitle(f"{par.upper()} coordinate check: L={args.layers}, base={args.base_width}, lr={args.lr}, "
                     f"B={args.B}, T={args.T}, seed={args.seed}")
        fig.tight_layout()
        fig.savefig(f"runs/b1_coord_check_{par}.png", dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base_width", type=int, default=64)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    data, (probe, _) = load_data(val_windows=8, T=args.T)
    res = {par: {str(w): coord_check(w, par, data, probe, args).tolist() for w in args.widths} for par in ("sp", "mup")}
    json.dump({"args": vars(args), "results": res}, open("runs/b1_coord_check.json", "w"))
    plot(res, args.widths, args)
    names = ["emb"] + [f"blk{i}" for i in range(args.layers)] + ["logits"]
    for par, r in res.items():
        r = np.array([r[str(w)] for w in args.widths])
        sl = [np.polyfit(np.log(args.widths), np.log(r[:, -1, j]), 1)[0] for j in range(r.shape[2])]
        # Step 2, not step 1: a zero readout passes no gradient to the network at step 1.
        sl2 = [np.polyfit(np.log(args.widths), np.log(r[:, 1, j]), 1)[0] for j in range(r.shape[2])]
        print(par, "slope at step 2:", dict(zip(names, np.round(sl2, 2).tolist())))
        print(par, f"slope at step {args.steps}:", dict(zip(names, np.round(sl, 2).tolist())))


if __name__ == "__main__":
    main()
