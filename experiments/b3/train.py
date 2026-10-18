"""B3 Task 5. Tier 0 training (AdamW or Muon) on the shared Trainer, with a CurvatureProbe on a fixed batch.

python -m experiments.b3.train --opt adamw --name b3_adamw
python -m experiments.b3.train --opt muon --name b3_muon
python -m experiments.b3.train --plot b3_adamw b3_muon
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lm.analysis.curvature import CurvatureProbe
from lm.data.sources import tokendata_source
from lm.model import Config, round_up
from lm.train.trainer import TrainConfig, Trainer


def train(args) -> None:
    train_tok, val, bpt, vocab = tokendata_source(args.data, args.val_windows, args.T, "cuda")
    cfg = Config(vocab_size=round_up(vocab, 64), d_model=args.d_model, n_layers=args.layers, n_heads=args.heads,
                 n_kv_heads=args.kv_heads, max_seq_len=args.T)
    steps = args.steps or args.tokens // (args.B * args.T)
    # The A3 recipe: warm-up 100, cosine to 0. Then 2 / eta(t) and 38 / eta(t) grow at the end of the run.
    tc = TrainConfig(name=args.name, B=args.B, T=args.T, steps=steps, optimizer=args.opt, lr=args.lr,
                     muon_lr=args.muon_lr, warmup=min(100, max(1, steps // 10)), min_lr_frac=0.0,
                     eval_every=args.curv_every, ckpt_every=0, log_every=args.curv_every, seed=args.seed)
    # A fixed curvature batch from the train split, kept on the GPU for the whole run.
    rng = np.random.default_rng(10_000 + args.seed)
    starts = rng.integers(0, len(train_tok) - args.T - 1, args.curv_B)
    buf = np.stack([train_tok[s : s + args.T + 1] for s in starts]).astype(np.int64)
    buf = torch.from_numpy(buf).cuda()
    probe = CurvatureProbe(buf[:, :-1], buf[:, 1:], every=args.curv_every, k=args.k, hutch=args.hutch,
                           chunk=args.curv_chunk)
    tr = Trainer(cfg, tc, train_tok, val, bpt, callbacks=[probe])
    tr.log_path.unlink(missing_ok=True)
    tr.fit()
    out = Path("runs") / f"{args.name}_curvature.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in probe.records))
    parts = {k[2:]: sum(r.get(k, 0.0) for r in probe.records) for k in probe.records[0] if k.startswith("t_")}
    print(f"train {tr.train_time:.0f}s, probe {probe.time_s:.0f}s ({100 * probe.time_s / (tr.train_time + probe.time_s):.0f} %"
          f" of training + probe); by part: " + ", ".join(f"{k} {v:.1f}s" for k, v in parts.items()))


def plot(names: list[str], path: Path) -> None:
    logs = [[json.loads(l) for l in open(Path("runs") / f"{n}_curvature.jsonl")] for n in names]
    fig, axes = plt.subplots(len(logs), 3, figsize=(14, 3.6 * len(logs)), squeeze=False)
    for i, (name, log) in enumerate(zip(names, logs)):
        s = np.array([r["step"] for r in log])
        lr = np.array([r["lrs"][-1] for r in log])  # the AdamW lr (Muon runs: AdamW drives the non-hidden tensors)
        ok = lr > 0
        a = axes[i, 0]
        a.semilogy(s, [r["eig_H"][0] for r in log], "o-", ms=3, label="lambda_max(H)")
        a.semilogy(s, [r["eig_G"][0] for r in log], "s-", ms=3, label="lambda_max(G)")
        a.semilogy(s[ok], 2 / lr[ok], "k--", lw=1, label="2 / eta(t)")
        a.set_title(f"{name}: raw sharpness"); a.legend(fontsize=7)
        a = axes[i, 1]
        if all("precond_sharp" in r for r in log[1:]):
            a.semilogy(s[1:], [r["precond_sharp"] for r in log[1:]], "o-", ms=3, label="lambda_max(P^-1 H)")
            a.semilogy(s[ok], 38 / lr[ok], "k--", lw=1, label="38 / eta(t)")
            a.legend(fontsize=7)
        a.set_title("preconditioned sharpness (AdamW only)")
        a = axes[i, 2]
        a.plot(s, [r["update_ratio"] for r in log], "o-", ms=3, label="u^T H u / (-g^T u)")
        a.axhline(2, color="k", ls="--", lw=1, label="2")
        a.set_title("sharpness along the update"); a.legend(fontsize=7)
        for a in axes[i]:
            a.set_xlabel("step")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="b3_adamw")
    ap.add_argument("--data", default="data/fineweb")
    ap.add_argument("--opt", default="adamw", choices=["adamw", "muon"])
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--kv_heads", type=int, default=2)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--muon_lr", type=float, default=0.02)
    ap.add_argument("--tokens", type=int, default=25_000_000)
    ap.add_argument("--steps", type=int, default=0, help="overrides tokens // (B T)")
    ap.add_argument("--val_windows", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--curv_every", type=int, default=200)
    ap.add_argument("--curv_B", type=int, default=4, help="sequences in the fixed curvature batch")
    ap.add_argument("--curv_chunk", type=int, default=2)
    ap.add_argument("--k", type=int, default=10, help="Lanczos iterations")
    ap.add_argument("--hutch", type=int, default=5, help="Hutchinson probes")
    ap.add_argument("--plot", nargs="*", default=None, help="only plot these run names")
    return ap


if __name__ == "__main__":
    args = parser().parse_args()
    if args.plot:
        plot(args.plot, Path("runs") / "b3_sharpness.png")
    else:
        train(args)
