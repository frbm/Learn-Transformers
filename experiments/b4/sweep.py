"""B4 Tasks 1, 4, 5. FLOP table, the resumable IsoFLOP sweep (μP, one base learning rate), and the Tier 1 check run."""
import argparse
import json
import time
from pathlib import Path

import torch

from lm.analysis.scaling import flops_per_token, param_counts
from lm.data.tokens import load_data
from lm.model.mup import make_model, mup_config, param_groups
from lm.train.trainer import train_run

BUDGETS = [1e14, 3e14, 1e15, 3e15]
SHAPES = [(128, 4), (192, 4), (256, 4), (384, 6), (512, 6)]


def n_ctx(T: int) -> float:
    """Mean keys per query under a causal kernel that skips masked blocks: (T + 1) / 2."""
    return (T + 1) / 2


def plan(C: float, d: int, L: int, args, vocab: int) -> dict:
    cfg = mup_config(d, L, "mup", vocab, args.T, args.base_width)
    fpt = flops_per_token(cfg, n_ctx(args.T))
    steps = max(1, round(C / fpt / (args.B * args.T)))
    tokens = steps * args.B * args.T
    return {"C": C, "d": d, "L": L, "seed": args.seed, **{f"N_{k}": v for k, v in param_counts(cfg).items()},
            "flops_per_token": fpt, "steps": steps, "tokens": tokens, "C_exact": fpt * tokens,
            "ratio_6N_total": fpt / (6 * param_counts(cfg)["total"]),
            "ratio_6N_nonemb": fpt / (6 * param_counts(cfg)["non_embedding"])}


def run_one(p: dict, args, data, val) -> dict:
    model = make_model(p["d"], p["L"], "mup", data.meta["vocab_size"], args.T, args.base_width, seed=p["seed"])
    opts = [torch.optim.AdamW(param_groups(model, args.lr, args.wd), betas=(0.9, 0.95), fused=True)]
    t0 = time.perf_counter()
    # The cosine schedule ends exactly at this run's own token count.
    hist, _ = train_run(model, opts, data, val, B=args.B, T=args.T, steps=p["steps"], seed=p["seed"],
                    warmup=max(1, min(100, p["steps"] // 10)), eval_every=max(p["steps"] // 4, 1))
    return p | {"val_loss": hist[-1]["val_loss"], "val_bpb": data.to_bpb(hist[-1]["val_loss"]),
                "epochs": p["tokens"] / data.split, "wall_s": time.perf_counter() - t0, "lr": args.lr,
                "base_width": args.base_width}


def key(r: dict) -> tuple:
    return (float(r["C"]), r["d"], r["L"], r["seed"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/fineweb")
    ap.add_argument("--out", default="runs/b4_sweep.jsonl")
    ap.add_argument("--budgets", type=float, nargs="+", default=BUDGETS)
    ap.add_argument("--shapes", type=int, nargs="+", default=[x for s in SHAPES for x in s], help="d L d L ...")
    ap.add_argument("--lr", type=float, default=2**-8, help="B1's μP base learning rate at base_width")
    ap.add_argument("--base_width", type=int, default=256)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--val_windows", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--table", action="store_true", help="print the Task 1 FLOP table and exit")
    ap.add_argument("--tier1", action="store_true", help="Task 5: d=512, L=8, --tokens tokens, own log")
    ap.add_argument("--tokens", type=float, default=200e6)
    args = ap.parse_args()
    shapes = list(zip(args.shapes[::2], args.shapes[1::2]))
    vocab = json.loads(Path(f"{args.data}_meta.json").read_text())["vocab_size"]

    if args.table:
        print(f"{'d':>4} {'L':>2} {'N_total':>10} {'N_nonemb':>10} {'flops/tok':>10} {'/6N_tot':>8} {'/6N_ne':>8}")
        for d, L in shapes:
            p = plan(1.0, d, L, args, vocab)
            print(f"{d:4d} {L:2d} {p['N_total']:10,d} {p['N_non_embedding']:10,d} {p['flops_per_token']:10.3e} "
                  f"{p['ratio_6N_total']:8.3f} {p['ratio_6N_nonemb']:8.3f}")
        return

    data, val = load_data(args.data, args.val_windows, args.T)
    if args.tier1:
        p = plan(1.0, 512, 8, args, vocab)
        p |= {"C": p["flops_per_token"] * args.tokens, "steps": round(args.tokens / (args.B * args.T))}
        p |= {"tokens": p["steps"] * args.B * args.T}
        p["C_exact"] = p["flops_per_token"] * p["tokens"]
        rec = run_one(p, args, data, val)
        print(json.dumps(rec))
        Path("runs/b4_tier1.json").write_text(json.dumps(rec, indent=1))
        return

    out = Path(args.out)
    # One line per finished run. A restart skips every key already on disk.
    done = {key(json.loads(l)) for l in out.read_text().splitlines()} if out.exists() else set()
    for C in args.budgets:
        for d, L in shapes:
            p = plan(C, d, L, args, vocab)
            if key(p) in done:
                continue
            rec = run_one(p, args, data, val)
            print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}), flush=True)
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
