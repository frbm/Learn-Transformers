"""C1 Task 3 and Task 6: the Tier 1 run on the shared Trainer.

python -m experiments.c1.train --name c1_tier1                              # d=512, L=8, T=1024, 200 M tokens
python -m experiments.c1.train --name c1_smoke --steps 20 --eval_every 10   # smoke run
python -m experiments.c1.train --name c1_tier1 --resume                     # exact resume from runs/c1_tier1/ckpt.pt
"""
import argparse
import json
from dataclasses import asdict

import torch

from lm.data.sources import tokendata_source, tokenfile_source
from lm.model import Config, round_up
from lm.tokenizer.bpe import Tokenizer
from lm.train.memory import predict_peak_memory, roofline_step_time
from lm.train.trainer import TrainConfig, Trainer
from lm.utils.bench import load_peaks

PROMPTS = ["The history of", "In this lesson, students will", "Water is"]


@torch.no_grad()
def samples(model, tok: Tokenizer, prompts: list[str], n_new: int = 60) -> list[str]:
    model.eval()
    out = []
    for p in prompts:
        ids = torch.tensor([tok.encode(p)], device=model.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=model.device.type == "cuda"):
            g = model.generate(ids, n_new)
        out.append(tok.decode(g[0].tolist()).decode("utf-8", errors="replace"))
    return out


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="c1_tier1")
    ap.add_argument("--data", default="data/fineweb", help="A3 TokenData prefix")
    ap.add_argument("--tokfile", default="", help="C2 token-file prefix (overrides --data)")
    ap.add_argument("--tokenizer", default="data/fineweb_tok.json")
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv_heads", type=int, default=4)
    ap.add_argument("--checkpointing", action="store_true")
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=200_000_000)
    ap.add_argument("--steps", type=int, default=0, help="overrides --tokens")
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "muon"])
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--muon_lr", type=float, default=0.02)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--schedule", default="cosine", choices=["cosine", "wsd"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--ce_chunk", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--val_windows", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop_at", type=int, default=0, help="stop (as if killed) after this step; for resume tests")
    return ap


def main():
    args = parser().parse_args()
    src = tokenfile_source if args.tokfile else tokendata_source
    train_tok, val, bpt, vocab = src(args.tokfile or args.data, args.val_windows, args.T, "cuda")
    cfg = Config(vocab_size=round_up(vocab, 64), d_model=args.d_model, n_layers=args.layers, n_heads=args.heads,
                 n_kv_heads=args.kv_heads, max_seq_len=args.T, checkpointing=args.checkpointing)
    steps = args.steps or args.tokens // (args.B * args.T * args.K)
    tc = TrainConfig(name=args.name, B=args.B, T=args.T, K=args.K, steps=steps, optimizer=args.optimizer,
                     lr=args.lr, muon_lr=args.muon_lr, warmup=min(args.warmup, max(1, steps // 10)),
                     schedule=args.schedule, compile=args.compile, ce_chunk=args.ce_chunk or None,
                     eval_every=args.eval_every, ckpt_every=args.ckpt_every, seed=args.seed)
    tr = Trainer(cfg, tc, train_tok, val, bpt)
    P, beta = load_peaks()
    # Write the prediction before the run. Then the run tests it.
    pred = {"pred_peak_gib": predict_peak_memory(cfg, args.B, args.T, args.optimizer, args.checkpointing,
                                                 tc.ce_chunk) / 2**30,
            "roofline_step_s": roofline_step_time(cfg, args.B, args.T, args.K, P, beta),
            "params_M": sum(p.numel() for p in tr.model.parameters()) / 1e6, "steps": steps, "P": P, "beta": beta,
            "flops_per_step": tr.flops_per_step, "epochs": steps * args.B * args.T * args.K / len(train_tok)}
    print(json.dumps(pred))
    if args.resume:
        tr.load()
        print(f"resumed at step {tr.step}")
    else:
        tr.log_path.unlink(missing_ok=True)
        tr.log({"prediction": pred, "cfg": asdict(cfg), "tc": asdict(tc)})
    tr.fit(until=args.stop_at or None)
    if args.stop_at:
        tr.save()
        return
    torch.save({"cfg": asdict(cfg), "model": tr.model.state_dict()}, tr.dir / "model.pt")
    for s in samples(tr.model, Tokenizer.load(args.tokenizer), PROMPTS):
        print("---\n" + s)


if __name__ == "__main__":
    main()
