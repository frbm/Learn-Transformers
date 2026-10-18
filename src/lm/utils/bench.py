"""Timing on the GPU: CUDA-event timers, peak matmul throughput P and copy bandwidth beta (C1 Task 1)."""
import json
import subprocess
import warnings
from pathlib import Path

import torch

SPEC_P, SPEC_BETA = 71e12, 504e9  # whitepaper scaling (C1 §4)
BENCH_FILE = Path("runs/c1_bench.json")


def cuda_time(fn, iters: int = 20, warmup: int = 3) -> float:
    """Seconds per call. Synchronise before and after: CUDA calls only queue work."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1e3 / iters


def matmul_tflops(n: int, dtype=torch.bfloat16, iters: int = 50) -> float:
    """Achieved TFLOP/s of an n x n x n matmul. Inputs and output are allocated once, outside the timed call."""
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    c = torch.empty(n, n, device="cuda", dtype=dtype)
    t = cuda_time(lambda: torch.matmul(a, b, out=c), iters=iters, warmup=max(3, iters // 10))
    return 2 * n**3 / t / 1e12


def copy_bandwidth(nbytes: int = 2**30, iters: int = 20) -> float:
    """GB/s of dst.copy_(src). A copy reads nbytes and writes nbytes."""
    src = torch.empty(nbytes, device="cuda", dtype=torch.uint8)
    dst = torch.empty_like(src)
    return 2 * nbytes / cuda_time(lambda: dst.copy_(src), iters=iters) / 1e9


def gpu_busy() -> int:
    """GPU utilisation in percent before the run. Another process on the card corrupts every number here."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True)
    return int(out.stdout.split()[0]) if out.returncode == 0 else -1


def load_peaks() -> tuple[float, float]:
    """Measured (P, beta) in FLOP/s and B/s from Task 1, else the derived values.

    Judge the measurement, not the conditions. nvidia-smi utilisation is a sampled duty cycle, and the desktop
    alone shows 10-20 %. A shared GPU shows up in the result: a matmul peak far below the derivation.
    """
    if BENCH_FILE.exists():
        d = json.loads(BENCH_FILE.read_text())
        P, beta = d["P_bf16_tflops"] * 1e12, d["beta_gbs"] * 1e9
        if not d.get("smoke") and P >= 0.7 * SPEC_P:
            return P, beta
        warnings.warn(f"{BENCH_FILE}: P = {P / 1e12:.0f} TFLOP/s (smoke={d.get('smoke')}) is not a valid peak. "
                      f"MFU uses the derived {SPEC_P / 1e12:.0f} TFLOP/s. Rerun experiments.c1.bench on an idle GPU.")
    return SPEC_P, SPEC_BETA
