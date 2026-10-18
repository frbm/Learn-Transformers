"""C1 Task 1. Measure the peak matmul throughput P and the copy bandwidth beta of this card.

python -m experiments.c1.bench            # full sweep, plot to runs/c1_roofline.png, numbers to runs/c1_bench.json
python -m experiments.c1.bench --smoke    # n <= 2048, few iterations
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lm.utils.bench import BENCH_FILE, SPEC_BETA, SPEC_P, copy_bandwidth, gpu_busy, matmul_tflops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--repeats", type=int, default=3, help="median over this many runs per size")
    args = ap.parse_args()
    busy = gpu_busy()
    if busy > 5:
        print(f"warning: the GPU is {busy}% busy before the run. The numbers are lower bounds.")
    sizes = [256, 512, 1024, 2048] if args.smoke else [256, 512, 1024, 2048, 4096, 8192]
    iters = 10 if args.smoke else 50

    def med(f):
        return float(np.median([f() for _ in range(args.repeats)]))

    res = {"bf16": [], "fp32": [], "tf32": []}
    for n in sizes:
        res["bf16"].append(med(lambda: matmul_tflops(n, torch.bfloat16, iters)))
        torch.backends.cuda.matmul.allow_tf32 = False
        res["fp32"].append(med(lambda: matmul_tflops(n, torch.float32, max(3, iters // 5))))
        torch.backends.cuda.matmul.allow_tf32 = True
        res["tf32"].append(med(lambda: matmul_tflops(n, torch.float32, max(3, iters // 5))))
        torch.backends.cuda.matmul.allow_tf32 = False
        print(f"n={n:5d}  bf16 {res['bf16'][-1]:6.1f}  fp32 {res['fp32'][-1]:6.1f}  tf32 {res['tf32'][-1]:6.1f} TFLOP/s")
    beta = med(lambda: copy_bandwidth(2**28 if args.smoke else 2**30, 5 if args.smoke else 20))
    P = max(res["bf16"])
    print(f"P_meas = {P:.1f} TFLOP/s ({P / (SPEC_P / 1e12):.0%} of 71), beta_meas = {beta:.0f} GB/s "
          f"({beta / (SPEC_BETA / 1e9):.0%} of 504), ridge = {P * 1e3 / beta:.0f} FLOP/B")

    Path("runs").mkdir(exist_ok=True)
    out = {"sizes": sizes, **res, "P_bf16_tflops": P, "beta_gbs": beta, "smoke": args.smoke, "gpu_util_before": busy,
           "device": torch.cuda.get_device_name()}
    (BENCH_FILE.with_name("c1_bench_smoke.json") if args.smoke else BENCH_FILE).write_text(json.dumps(out, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = np.logspace(np.log10(sizes[0]), np.log10(sizes[-1]), 100)
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, m in (("bf16", "o-"), ("tf32", "s-"), ("fp32", "^-")):
        ax.plot(sizes, res[k], m, label=k)
    # Proposition 7 with M = d_in = d_out = n and 2-byte elements: I = n / 3.
    ax.plot(ns, np.minimum(P, ns / 3 * beta / 1e3), "k--", label=f"roofline min(P, I beta), I = n/3")
    ax.axhline(SPEC_P / 1e12, color="gray", ls=":", label="71 TFLOP/s (derived)")
    ax.set(xscale="log", xlabel="n (n x n x n matmul)", ylabel="TFLOP/s", title=torch.cuda.get_device_name())
    ax.legend()
    fig.tight_layout()
    fig.savefig("runs/c1_roofline_smoke.png" if args.smoke else "runs/c1_roofline.png", dpi=120)


if __name__ == "__main__":
    main()
