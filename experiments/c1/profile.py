"""C1 Task 5. Profile training steps of the Tier 1 configuration: top kernels, matmul share, idle gaps.

python -m experiments.c1.profile                 # eager, then compiled if Triton exists
python -m experiments.c1.profile --B 4           # smaller micro-batch (smoke)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, schedule

from lm.data.sources import tokendata_source
from lm.train.trainer import Trainer, TrainConfig, compile_available
from lm.model import Config

MATMUL = ("gemm", "cutlass", "sm89", "sm80", "cublas", "matmul")


def kernels(trace_path: str) -> list[tuple[float, float, str]]:
    """(start, end, name) of every CUDA kernel in a Chrome trace, in time order."""
    ev = json.loads(Path(trace_path).read_text())["traceEvents"]
    return sorted((e["ts"], e["ts"] + e["dur"], e["name"]) for e in ev if e.get("cat") == "kernel" and "dur" in e)


def summarise(k: list[tuple[float, float, str]], steps: int) -> dict:
    """Kernel time by name, matmul share, and the idle fraction of the GPU between first and last kernel."""
    by: dict[str, float] = {}
    for s, e, n in k:
        by[n] = by.get(n, 0.0) + e - s
    total = sum(by.values())
    mm = sum(t for n, t in by.items() if any(m in n.lower() for m in MATMUL))
    busy, end, gaps = 0.0, k[0][0], []
    for s, e, _ in k:
        if s > end:
            gaps.append(s - end)
        busy += max(0.0, e - max(s, end))
        end = max(end, e)
    top = sorted(by.items(), key=lambda r: -r[1])[:3]
    return {"gpu_ms_per_step": total / steps / 1e3, "matmul_share": mm / total,
            "idle_fraction": 1 - busy / (end - k[0][0]), "median_gap_us": float(np.median(gaps)) if gaps else 0.0,
            "top3": [(n[:90], t / total) for n, t in top]}


def run(tr: Trainer, tag: str, out: Path) -> dict:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=schedule(wait=1, warmup=2, active=3), record_shapes=True) as prof:
        for _ in range(6):
            tr.train_step()
            prof.step()
    torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=15))
    trace = str(out / f"c1_trace_{tag}.json")
    prof.export_chrome_trace(trace)
    res = {"tag": tag, **summarise(kernels(trace), steps=3)}
    print(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--ce_chunk", type=int, default=0)
    args = ap.parse_args()
    out = Path("runs")
    train_tok, val, bpt, vocab = tokendata_source("data/fineweb", 4, args.T, "cuda")
    cfg = Config(vocab_size=16448, d_model=512, n_layers=8, n_heads=8, n_kv_heads=4, max_seq_len=args.T)
    results = []
    for comp in (False, True):
        if comp and not compile_available()[0]:
            print(compile_available()[1])
            continue
        tc = TrainConfig(name="c1_profile", B=args.B, T=args.T, steps=10, warmup=1, compile=comp,
                         ce_chunk=args.ce_chunk or None)
        results.append(run(Trainer(cfg, tc, train_tok, val, bpt), "compiled" if comp else "eager", out))
    (out / "c1_profile.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
