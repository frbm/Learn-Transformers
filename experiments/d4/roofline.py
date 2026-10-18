"""D4 Task 2: decode throughput and ITL against batch size, with the Proposition 7 bound.

python -m experiments.d4.roofline --ckpt runs/c1/tier1.pt          # full sweep, B = 1 .. 256, ctx 128 and 1024
python -m experiments.d4.roofline --smoke                          # random Tier 1 shape, 3 batch sizes
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from lm.verse.common import load_model
from lm.infer.generate import decode_bound, kv_cache_bytes, time_generation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--B", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--ctx", type=int, nargs="+", default=[128, 1024])
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--P", type=float, default=71e12, help="peak bf16 FLOP/s (C1 measured value)")
    ap.add_argument("--beta", type=float, default=504e9, help="bandwidth B/s (C1 measured value)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="runs/d4_roofline")
    a = ap.parse_args()
    if a.smoke:
        a.B, a.ctx, a.new = [1, 8, 32], [128], 16
    model = load_model(a.ckpt, a.vocab, max_seq_len=max(a.ctx) + a.new + 2)
    cfg = model.cfg
    N = sum(p.numel() for p in model.parameters())
    W = N * 2
    rows = []
    for ctx in a.ctx:
        for B in a.B:
            K = kv_cache_bytes(cfg.n_layers, cfg.n_kv_heads, cfg.d_head, ctx + a.new // 2)
            for graph in (False, True):
                try:
                    r = time_generation(model, B, ctx, a.new, cuda_graph=graph)
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    continue
                t = decode_bound(B, W, N, K, a.P, a.beta)
                rows.append({"ctx": ctx, "B": B, "graph": graph, **r, "bound_itl_ms": 1e3 * t,
                             "bound_tok_per_s": B / t, "peak_gib": torch.cuda.max_memory_allocated() / 2**30})
                print(json.dumps(rows[-1]), flush=True)
                torch.cuda.reset_peak_memory_stats()
    out = Path(a.out)
    out.parent.mkdir(exist_ok=True)
    out.with_suffix(".jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for ctx in a.ctx:
        for graph, ls in ((False, "-o"), (True, "-s")):
            rs = [r for r in rows if r["ctx"] == ctx and r["graph"] == graph]
            if not rs:
                continue
            lab = f"ctx {ctx}, {'CUDA graph' if graph else 'eager'}"
            ax[0].plot([r["B"] for r in rs], [r["tok_per_s"] for r in rs], ls, label=lab)
            ax[1].plot([r["B"] for r in rs], [r["itl_ms"] for r in rs], ls, label=lab)
        rs = [r for r in rows if r["ctx"] == ctx and not r["graph"]]
        ax[0].plot([r["B"] for r in rs], [r["bound_tok_per_s"] for r in rs], "k--", alpha=0.5, label=f"bound ctx {ctx}")
        ax[1].plot([r["B"] for r in rs], [r["bound_itl_ms"] for r in rs], "k--", alpha=0.5)
    for x, yl in zip(ax, ("decode tokens/s", "ITL (ms)")):
        x.set_xscale("log", base=2); x.set_yscale("log"); x.set_xlabel("batch B"); x.set_ylabel(yl)
    ax[0].legend(fontsize=7)
    fig.suptitle(f"{N / 1e6:.1f} M params, W = {W / 1e6:.0f} MB, P = {a.P / 1e12:.0f} TFLOP/s, beta = {a.beta / 1e9:.0f} GB/s")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=120)


if __name__ == "__main__":
    main()
