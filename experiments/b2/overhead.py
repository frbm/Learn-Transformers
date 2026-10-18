"""B2 Task 6. Newton-Schulz overhead: step time with ns_steps=5 vs ns_steps=0 (identity), CUDA events."""
import argparse
import json

import torch
import torch.nn.functional as F

from lm.model.mup import make_model
from lm.model.mup import classify
from lm.train.optim import build_optimizers


def ns_flops(model, steps: int = 5) -> float:
    total = 0.0
    for n, p in model.named_parameters():
        if classify(n, p) == "hidden":
            m, k = sorted(p.shape)
            total += steps * (4 * m * m * k + 2 * m**3)
    return total


def train_flops(model, tokens: int, T: int) -> float:
    """6 N tau for the matmul weights, plus causal attention (QK^T and PV, half the pairs)."""
    cfg = model.cfg
    n_mat = sum(p.numel() for n, p in model.named_parameters() if classify(n, p) in ("hidden", "readout"))
    return 6 * tokens * n_mat + 6 * cfg.n_layers * tokens * T * cfg.d_model


def time_step(d: int, tokens: int, ns_steps: int, args) -> tuple[float, float, float]:
    model = make_model(d, args.layers, "mup", args.vocab, args.T, base_width=d, seed=0)
    opts = build_optimizers(model, 0.02, 3e-3, 0.0, ns_steps=ns_steps)
    micro = min(tokens, args.micro_tokens) // args.T
    n_micro = tokens // (micro * args.T)
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randint(0, args.vocab, (micro, args.T + 1), device="cuda", generator=g)

    def step():
        for _ in range(n_micro):                                    # gradient accumulation to reach tau tokens
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(x[:, :-1]).flatten(0, 1).float(), x[:, 1:].flatten()) / n_micro
            loss.backward()
        for o in opts:
            o.step()
        for o in opts:
            o.zero_grad(set_to_none=True)

    for _ in range(args.warmup):
        step()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(args.iters):
        step()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / args.iters
    nsf, trf = ns_flops(model), train_flops(model, tokens, args.T)
    del model, opts
    torch.cuda.empty_cache()
    return ms, nsf, trf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[256, 512, 1024])
    ap.add_argument("--tokens", type=int, nargs="+", default=[8192, 16384, 65536])
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--micro_tokens", type=int, default=8192, help="tokens per forward; caps activation memory")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    rows = []
    print(f"{'d':>5} {'tokens':>6} {'ms NS':>8} {'ms id':>8} {'overhead':>9} {'5d/tau':>7} {'NS/train FLOPs':>15}")
    for d in args.dims:
        for tau in args.tokens:
            t_ns, nsf, trf = time_step(d, tau, 5, args)
            t_id, _, _ = time_step(d, tau, 0, args)
            r = {"d": d, "tokens": tau, "ms_ns": t_ns, "ms_id": t_id, "overhead": t_ns / t_id - 1, "pred_5d_tau": 5 * d / tau,
                 "pred_flops": nsf / trf}
            rows.append(r)
            print(f"{d:5d} {tau:6d} {t_ns:8.2f} {t_id:8.2f} {r['overhead']:9.1%} {r['pred_5d_tau']:7.1%} {r['pred_flops']:15.1%}", flush=True)
    json.dump({"args": vars(args), "gpu": torch.cuda.get_device_name(), "rows": rows}, open("runs/b2_overhead.json", "w"))


if __name__ == "__main__":
    main()
