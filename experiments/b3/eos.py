"""B3 Task 4. Full-batch gradient descent on fixed validation windows. Track lambda_max(H) against 2 / eta."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from lm.analysis.curvature import Curvature, FlatModel, top_eigs
from lm.data.tokens import TokenData
from lm.model import GPT, Config, round_up


def run(args, lr: float, x, y, vocab: int) -> list[dict]:
    torch.manual_seed(args.seed)
    cfg = Config(vocab_size=round_up(vocab, 64), d_model=args.d_model, n_layers=args.layers, n_heads=args.heads,
                 n_kv_heads=args.kv_heads, max_seq_len=args.T, pos_encoding=args.pos)
    model = GPT(cfg).cuda().eval()
    fm = FlatModel(model)
    curv = Curvature(fm, x, y, args.chunk, cache=args.cache)
    # Gradient descent in flat-parameter space. No optimiser object, no .grad buffers.
    theta = fm.flat()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    t_train = t_curv = 0.0
    log, eta = [], lr
    for step in range(args.steps + 1):
        if args.halve_at and step == args.halve_at:
            eta = lr / 2
        ev[0].record()
        loss, g = curv.value_and_grad(theta)
        ev[1].record()
        rec = {"lr": lr, "step": step, "eta": eta, "loss": loss, "grad_norm": g.norm().item()}
        if step % args.every == 0:
            ev[2].record()
            # The math backend supports double backward. Flash and memory-efficient kernels do not.
            with sdpa_kernel(SDPBackend.MATH):
                rec["lam_max"] = top_eigs(lambda v: curv.hvp(theta, v), fm.dim, args.k, 1, "cuda", seed=step)[0]
            curv.release()
            ev[3].record(); torch.cuda.synchronize()
            t_curv += ev[2].elapsed_time(ev[3])
            print(json.dumps(rec), flush=True)
        torch.cuda.synchronize()
        t_train += ev[0].elapsed_time(ev[1])
        log.append(rec)
        if not np.isfinite(loss) or loss > 2 * log[0]["loss"]:
            print(f"lr={lr}: diverged at step {step}")
            break
        theta.sub_(g, alpha=eta)
    print(f"lr={lr}: train {t_train / 1e3:.1f}s, curvature {t_curv / 1e3:.1f}s, "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    return log


def plot(logs: list[list[dict]], path: Path) -> None:
    fig, axes = plt.subplots(2, len(logs), figsize=(4.5 * len(logs), 6), sharex="col", squeeze=False)
    for j, log in enumerate(logs):
        s = np.array([r["step"] for r in log])
        axes[0, j].plot(s, [r["loss"] for r in log], lw=1)
        axes[0, j].set_title(f"eta = {log[0]['lr']:g}")
        m = [r for r in log if "lam_max" in r]
        axes[1, j].plot([r["step"] for r in m], [r["lam_max"] for r in m], "o-", ms=3, label="lambda_max(H)")
        axes[1, j].plot(s, [2 / r["eta"] for r in log], "k--", lw=1, label="2 / eta")
        axes[1, j].set_xlabel("step")
    axes[0, 0].set_ylabel("full-batch loss (nats)")
    axes[1, 0].set_ylabel("sharpness")
    axes[1, 0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/fineweb")
    ap.add_argument("--lrs", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--halve_at", type=int, default=0, help="halve eta at this step (0 = never)")
    ap.add_argument("--n_seq", type=int, default=512)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=64, help="sequences per micro-batch")
    ap.add_argument("--cache", action="store_true", help="keep per-chunk graphs; needs memory for all chunks")
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--kv_heads", type=int, default=1)
    ap.add_argument("--pos", default="rope", choices=["nope", "rope"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="b3_eos")
    args = ap.parse_args()

    data = TokenData(args.data)
    x, y = data.val_windows(args.n_seq, args.T)
    logs = [run(args, lr, x, y, data.meta["vocab_size"]) for lr in args.lrs]
    with open(Path("runs") / f"{args.name}.jsonl", "w") as f:
        for log in logs:
            f.writelines(json.dumps(r) + "\n" for r in log)
    plot(logs, Path("runs") / f"{args.name}.png")


if __name__ == "__main__":
    main()
