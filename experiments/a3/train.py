"""A3 Task 6 (and A4 Task 3). Minimal Tier 0 loop. C1 replaces it."""
import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lm.data.tokens import TokenData, evaluate
from lm.model import GPT, Config, round_up


def build_config(args, vocab: int) -> Config:
    return Config(vocab_size=round_up(vocab, 64), d_model=args.d_model, n_layers=args.layers, n_heads=args.heads,
                  n_kv_heads=args.kv_heads, max_seq_len=args.T, pos_encoding=args.pos,
                  use_residual=not args.no_residual, use_mlp=not args.no_mlp)


def lr_at(step: int, steps: int, peak: float, warmup: int = 100) -> float:
    return peak * min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * step / steps))


def train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data = TokenData(args.data)
    cfg = build_config(args, data.meta["vocab_size"])
    model = GPT(cfg).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    xv, yv = data.val_windows(args.val_windows, args.T)
    steps = args.tokens // (args.B * args.T)
    log = open(Path("runs") / f"{args.name}.jsonl", "w")
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, {steps} steps")

    t0, tok_seen = time.perf_counter(), 0
    for step in range(steps + 1):
        if step % args.eval_every == 0 or step == steps:
            vl = evaluate(model, xv, yv)
            dt = time.perf_counter() - t0
            rec = {"step": step, "val_loss": vl, "val_bpb": data.to_bpb(vl), "tok_per_s": tok_seen / max(dt, 1e-9),
                   "peak_gib": torch.cuda.max_memory_allocated() / 2**30}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
        if step == steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, args.lr)
        x, y = data.get_batch(args.B, args.T, rng=rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(x).flatten(0, 1).float(), y.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        tok_seen += x.numel()
    print(f"wall clock {time.perf_counter() - t0:.0f}s")
    if args.out:
        torch.save({"cfg": asdict(cfg), "model": model.state_dict()}, args.out)
    return model, data


def ngram_baseline(data: TokenData, train_bytes: int = 20_000_000) -> None:
    """A1's best n-gram, fitted on the bytes of the same text."""
    from lm.tokenizer.bpe import Tokenizer
    from lm.eval.metrics import bits_per_byte
    from lm.baselines.ngram import NGram

    tok = Tokenizer.load(data.meta.get("tokenizer", "data/fineweb_tok.json"))
    eot = tok.special["<|endoftext|>"]
    strip = lambda ids: tok.decode([i for i in ids if i != eot])
    tr = np.frombuffer(strip(data.data[: train_bytes // 4].tolist()), dtype=np.uint8).astype(np.int64)
    va = np.frombuffer(strip(data.data[data.split :].tolist()), dtype=np.uint8).astype(np.int64)
    for n in (4, 5, 6):
        m = NGram(n, 0.01).fit(tr)
        print(f"n-gram n={n}: val {bits_per_byte(m.nll(va), len(va) - n + 1):.3f} bpb (fitted on {len(tr) / 1e6:.0f} MB)")


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/fineweb")
    ap.add_argument("--name", default="a3_tier0")
    ap.add_argument("--out", default="")
    ap.add_argument("--pos", default="nope", choices=["nope", "rope"])
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--kv_heads", type=int, default=2)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--tokens", type=int, default=25_000_000)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--val_windows", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_residual", action="store_true")
    ap.add_argument("--no_mlp", action="store_true")
    ap.add_argument("--ngram", action="store_true")
    return ap


if __name__ == "__main__":
    args = parser().parse_args()
    _, data = train(args)
    if args.ngram:
        ngram_baseline(data)
