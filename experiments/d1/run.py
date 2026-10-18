"""D1 Tasks 1 and 3: SFT data statistics, the Proposition 3 check, the SFT run and the replay (lambda) sweep.

python -m experiments.d1.run --base runs/c1_tier1/model.pt                  # lambda in {0, 0.1, 0.3}
python -m experiments.d1.run --base ... --lambdas 0 --steps 30 --eval_every 10   # smoke run
"""
import argparse
import json
import math
from pathlib import Path

import torch

from lm.data.sft import SPECIALS, add_special_tokens, build_sft
from lm.data.sources import tokendata_source
from lm.model.checkpoint import load_model, save_model
from lm.tokenizer.bpe import Tokenizer
from lm.train.optim import adamw
from lm.train.sft import SFTEval, SFTObjective, train_sample
from lm.train.trainer import TrainConfig, Trainer


def load_base(path: str, device: str = "cuda"):
    return load_model(path, device)[0]


@torch.no_grad()
def kl_check(model, tok: Tokenizer, val: tuple, n: int = 100) -> dict:
    """Proposition 3 on n FineWeb contexts: KL(old || new restricted to old tokens) at every position."""
    model.eval()
    chunks = val[0][:n].split(10)
    # Keep fp64 log-probs on the CPU. The KL of interest (about 1e-4 nats) is below bf16 resolution.
    p = [model(c).double().log_softmax(-1).cpu() for c in chunks]
    V = p[0].shape[-1]
    add_special_tokens(model, tok, SPECIALS)
    kl, p_new = [], 0.0
    for c, pc in zip(chunks, p):
        q = model(c).double().log_softmax(-1).cpu()
        kl.append((pc.exp() * (pc - q[..., :V])).sum(-1))
        p_new = max(p_new, q[..., V:].exp().max().item())
    kl = torch.cat(kl)
    k = model.cfg.vocab_size - V
    return {"V": V, "k": k, "kl_max": kl.max().item(), "kl_mean": kl.mean().item(), "bound": math.log(1 + k / V),
            "p_new_max": p_new, "one_over_V": 1 / V}


def train_sft(model, params: list, sft: dict, args, log_dir: Path, tag: str, pre=None) -> list[dict]:
    """SFT on the shared Trainer. pre = (FineWeb train tokens, val windows, bytes per token) for replay and forgetting.

    params: the trainable parameters (all for full SFT, the adapters for LoRA).
    """
    train_tok, val, bpt, _ = pre or tokendata_source(args.data, args.val_windows, args.T_pre, "cuda")
    obj = SFTObjective(sft["train"], args.B, args.K, args.lam, args.weighting,
                       flex=None if args.attn == "flex" else False, seed=args.seed)
    steps = args.steps or args.epochs * obj.steps_per_epoch
    tc = TrainConfig(name=f"{log_dir.name}/{tag}", out_dir=str(log_dir.parent), B=args.B, T=args.T_pre, K=args.K,
                     steps=steps, lr=args.lr, warmup=max(1, int(args.warmup * steps)), min_lr_frac=0.0,
                     eval_every=10**9, ckpt_every=0, log_every=max(1, steps // 20), seed=args.seed)
    ev = SFTEval(sft["val"], train_sample(sft["train"], len(sft["val"]["input_ids"])),
                 every=args.eval_every or obj.steps_per_epoch, tag=tag,
                 extra={"lam": obj.lam, "weighting": args.weighting})
    opt = adamw(model, args.lr, args.wd) if params is None else         torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.wd, fused=True)
    tr = Trainer(model.cfg, tc, train_tok, val, bpt, model=model, opts=[opt], callbacks=[ev], objective=obj)
    tr.log_path.unlink(missing_ok=True)
    tr.fit(verbose=False)
    return ev.records


def frontier_plot(base: dict, finals: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter([base["val_loss"]], [base["fineweb_bpb"]], c="k", marker="x", label="base")
    for r in finals:
        ax.scatter(r["val_loss"], r["fineweb_bpb"], label=f"λ={r['lam']:.1f}")
    ax.set_xlabel("verse val loss (nats/token, response only)")
    ax.set_ylabel("FineWeb val bpb (forgetting)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="runs/c1_tier1/model.pt")
    ap.add_argument("--tok", default="data/fineweb_tok.json")
    ap.add_argument("--verse", default="data/verse.jsonl")
    ap.add_argument("--data", default="data/fineweb")
    ap.add_argument("--out", default="runs/d1")
    ap.add_argument("--T", type=int, default=1024, help="packed SFT row length: 10 % of poems are longer at 1024, 35 % at 512")
    ap.add_argument("--T_pre", type=int, default=512, help="replay and FineWeb evaluation window")
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--K", type=int, default=10, help="micro-batches per optimizer step")
    ap.add_argument("--lr", type=float, default=3e-4, help="about a tenth of the pretraining peak")
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=None, help="override the step count (smoke runs)")
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.1, 0.3])
    ap.add_argument("--weighting", default="token", choices=["token", "sequence"])
    ap.add_argument("--attn", default="flex", choices=["flex", "dense"])
    ap.add_argument("--val_windows", type=int, default=64)
    ap.add_argument("--eval_every", type=int, default=None, help="steps between evaluations (default: per epoch)")
    ap.add_argument("--seed", type=int, default=0)
    return ap


def main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pre = tokendata_source(args.data, args.val_windows, args.T_pre, "cuda")
    finals, base_rec = [], None
    for lam in args.lambdas:
        tok = Tokenizer.load(args.tok)
        model = load_base(args.base)
        kl = kl_check(model, tok, pre[1])
        sft, _, info = build_sft(tok, args.T, args.verse)
        if base_rec is None:
            print("Proposition 3:", kl)
            print(json.dumps(info, indent=1))
            (out / "data_stats.json").write_text(json.dumps({"kl_check": kl, **info}, indent=1))
        args.lam = lam
        hist = train_sft(model, None, sft, args, out, f"sft_lam{lam}", pre)
        (out / f"sft_lam{lam}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in hist))
        base_rec = base_rec or hist[0]
        finals.append(hist[-1])
        save_model(model, out / f"sft_lam{lam}.pt", special=list(tok.special))
    frontier_plot(base_rec, finals, out / "frontier.png")


if __name__ == "__main__":
    main(parser().parse_args())
