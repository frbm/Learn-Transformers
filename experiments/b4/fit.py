"""B4 Task 4 analysis and Task 5 bookkeeping. IsoFLOP parabolas (approach 2), parametric Huber fit with bootstrap
(approach 3) for both N conventions, a pre-registered Tier 1 prediction, then the check."""
import argparse
import datetime
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lm.analysis.scaling import NAMES, bootstrap_fits, fit_scaling_law

CONVENTIONS = ("N_total", "N_non_embedding")


def isoflop(runs: list[dict], conv: str, ax=None) -> dict:
    """Parabola in log N per budget. Its vertex is N*(C). Then a power law N* = k C^a."""
    by_c = defaultdict(list)
    for r in runs:
        by_c[r["C"]].append(r)
    Cs, Ns = [], []
    for C, rs in sorted(by_c.items()):
        x, y = np.log([r[conv] for r in rs]), np.array([r["val_loss"] for r in rs])
        if len(rs) < 3:
            continue
        c2, c1, c0 = np.polyfit(x, y, 2)
        if ax is not None:
            xs = np.linspace(x.min(), x.max(), 100)
            ax.plot(np.exp(x), y, "o", label=f"C = {C:.0e}")
            ax.plot(np.exp(xs), np.polyval([c2, c1, c0], xs), "-", color=ax.lines[-1].get_color(), lw=1)
        # A parabola that opens downwards has no minimum. Skip it rather than report a maximum.
        if c2 > 0:
            Cs.append(C); Ns.append(np.exp(-c1 / (2 * c2)))
    out = {"C": Cs, "N_star": Ns}
    if len(Cs) >= 2:
        out["a"], out["log_k"] = map(float, np.polyfit(np.log(Cs), np.log(Ns), 1))
    return out


def parametric(runs: list[dict], conv: str, n_boot: int, level: float) -> tuple[dict, list[dict]]:
    N, D, L = (np.array([r[k] for r in runs], dtype=float) for k in (conv, "tokens", "val_loss"))
    fit = fit_scaling_law(N, D, L)
    fits = bootstrap_fits(N, D, L, n_boot=n_boot)
    arr = np.array([[p[k] for k in NAMES] + [p["beta"] / (p["alpha"] + p["beta"])] for p in fits])
    q = np.quantile(arr, [(1 - level) / 2, (1 + level) / 2], axis=0)
    fit["a"] = fit["beta"] / (fit["alpha"] + fit["beta"])
    fit["ci"] = {k: q[:, j].tolist() for j, k in enumerate(NAMES + ("a",))}
    return fit, fits


def law(p: dict, N: float, D: float) -> float:
    return p["E"] + p["A"] * N ** -p["alpha"] + p["B"] * D ** -p["beta"]


def synthetic_runs() -> list[dict]:
    """A fake sweep on the planned shapes, from a known law. It exercises the pipeline without the GPU."""
    from experiments.b4.sweep import BUDGETS, SHAPES, plan
    a = argparse.Namespace(T=512, B=16, base_width=256, seed=0)
    rng = np.random.default_rng(0)
    runs = [plan(C, d, L, a, 16384) for C in BUDGETS for d, L in SHAPES]
    for r in runs:
        r["val_loss"] = (2.0 + 60 * r["N_total"] ** -0.3 + 300 * r["tokens"] ** -0.3) * np.exp(0.005 * rng.standard_normal())
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/b4_sweep.jsonl")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--level", type=float, default=0.95)
    ap.add_argument("--name", default="b4")
    ap.add_argument("--predict", action="store_true", help="write the Tier 1 prediction (never overwrites)")
    ap.add_argument("--check", action="store_true", help="compare runs/b4_tier1.json with the prediction")
    ap.add_argument("--tier1_result", default="runs/b4_tier1.json")
    ap.add_argument("--tier1_tokens", type=float, default=200e6)
    ap.add_argument("--data", default="data/fineweb", help="for the vocabulary size of the Tier 1 model")
    args = ap.parse_args()
    # A synthetic run never writes the real prediction file. The real one is written once and never replaced.
    if args.synthetic:
        args.name += "_synthetic"
    pred_path = Path(f"runs/{args.name}_prediction.json")

    if args.check:
        pred, got = json.loads(pred_path.read_text()), json.loads(Path(args.tier1_result).read_text())
        lo, hi = pred["interval"]
        miss = got["val_loss"] - pred["point"]
        print(f"predicted {pred['point']:.4f} [{lo:.4f}, {hi:.4f}] on {pred['date']}; got {got['val_loss']:.4f}; "
              f"miss {miss:+.4f} = {miss / (hi - lo):+.2f} interval widths")
        return

    runs = synthetic_runs() if args.synthetic else [json.loads(l) for l in open(args.runs)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    summary = {}
    for i, conv in enumerate(CONVENTIONS):
        iso = isoflop(runs, conv, axes[i])
        fit, fits = parametric(runs, conv, args.n_boot, args.level)
        summary[conv] = {"isoflop": iso, "parametric": fit}
        axes[i].set_xscale("log"); axes[i].set_xlabel(conv); axes[i].set_ylabel("final val loss (nats)")
        axes[i].set_title(f"IsoFLOP, {conv}: a = {iso.get('a', float('nan')):.3f}"); axes[i].legend(fontsize=7)
        print(f"{conv}: approach 2 a = {iso.get('a', float('nan')):.3f}; approach 3 a = {fit['a']:.3f} "
              f"[{fit['ci']['a'][0]:.3f}, {fit['ci']['a'][1]:.3f}]")
        for k in NAMES:
            print(f"   {k:>5} = {fit[k]:10.4g}   {int(100 * args.level)} % CI [{fit['ci'][k][0]:.4g}, {fit['ci'][k][1]:.4g}]")
        if conv == "N_total":
            ax = axes[2]
            if "a" in iso:
                ax.loglog(iso["C"], iso["N_star"], "o", label="IsoFLOP vertices")
                cs = np.logspace(np.log10(min(iso["C"])), np.log10(max(iso["C"])), 50)
                ax.loglog(cs, np.exp(iso["log_k"]) * cs ** iso["a"], "-", label=f"a = {iso['a']:.3f}")
            ax.set_xlabel("C (FLOPs)"); ax.set_ylabel("N* (total)"); ax.legend(fontsize=7)
            if args.predict:
                if pred_path.exists():
                    print(f"{pred_path} exists. A prediction is written once and never replaced.")
                else:
                    from experiments.b4.sweep import plan
                    vocab = json.loads(Path(f"{args.data}_meta.json").read_text())["vocab_size"]
                    N1 = plan(1.0, 512, 8, argparse.Namespace(T=512, B=16, base_width=256, seed=0), vocab)["N_total"]
                    preds = np.array([law(p, N1, args.tier1_tokens) for p in fits])
                    rec = {"date": datetime.date.today().isoformat(), "N_total": N1, "tokens": args.tier1_tokens,
                           "point": law(fit, N1, args.tier1_tokens), "level": args.level, "n_runs": len(runs),
                           "interval": np.quantile(preds, [(1 - args.level) / 2, (1 + args.level) / 2]).tolist(),
                           "synthetic": args.synthetic}
                    pred_path.write_text(json.dumps(rec, indent=1))
                    print("prediction:", rec)
    fig.tight_layout()
    fig.savefig(f"runs/{args.name}_isoflop.png", dpi=120)
    Path(f"runs/{args.name}_fit.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
