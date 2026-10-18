"""A3 Task 7: naive attention against SDPA. Time and peak memory, forward + backward."""
import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lm.model.attention import attention_core, attention_sdpa


def measure(fn, T, B=8, n_q=8, n_kv=8, d_h=64, dtype=torch.bfloat16, reps=3):
    mk = lambda h: torch.randn(B, h, T, d_h, device="cuda", dtype=dtype, requires_grad=True)
    q, k, v = mk(n_q), mk(n_kv), mk(n_kv)
    fn(q, k, v).sum().backward()  # warm-up: kernel selection and allocator
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(q, k, v).sum().backward()
    # GPU calls are asynchronous. Synchronise before you read the clock.
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps, (torch.cuda.max_memory_allocated() - base) / 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096, 8192])
    Ts = ap.parse_args().T
    res = {}
    for name, fn in (("naive", attention_core), ("sdpa", attention_sdpa)):
        res[name] = []
        for T in Ts:
            try:
                dt, mem = measure(fn, T)
            except torch.OutOfMemoryError:
                dt, mem = float("nan"), float("nan")
            torch.cuda.empty_cache()
            res[name].append((dt, mem))
            print(f"{name:>6} T={T:>5}: {dt * 1e3:8.2f} ms  {mem:6.2f} GiB", flush=True)
    print(f"prediction: fp32 scores alone = B*n_q*T^2*4 bytes > 12 GB at T = {np.sqrt(12e9 / (8 * 8 * 4)):.0f}")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for name, rows in res.items():
        r = np.array(rows)
        ax[0].loglog(Ts, r[:, 0], "o-", label=name)
        ax[1].loglog(Ts, r[:, 1], "o-", label=name)
        ok = ~np.isnan(r[:, 0])
        slope_t = np.polyfit(np.log(np.array(Ts)[ok][-3:]), np.log(r[ok, 0][-3:]), 1)[0]
        slope_m = np.polyfit(np.log(np.array(Ts)[ok][-3:]), np.log(r[ok, 1][-3:]), 1)[0]
        print(f"{name}: time slope {slope_t:.2f}, memory slope {slope_m:.2f} (last 3 points)")
    ax[0].set(xlabel="T", ylabel="fwd+bwd s"); ax[1].set(xlabel="T", ylabel="peak GiB")
    ax[0].legend(); fig.tight_layout(); fig.savefig("runs/a3_attention_bench.png", dpi=150)


if __name__ == "__main__":
    main()
