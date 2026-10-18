"""D2 Task 2 (rank sweep, standard vs rank-stabilised scaling) and the LoRA points of Task 3 (DoRA, PiSSA at r=16).

The SFT task and loop are D1's (`experiments.d1.run.train_sft`, on the shared Trainer). The embedding is frozen under LoRA, so the header is plain
text: new special tokens would stay at the mean embedding and all four would look the same.
"""
import json
from pathlib import Path

import torch

from lm.tokenizer.bpe import Tokenizer
from lm.data.sources import tokendata_source
from lm.model.lora import apply_lora, lora_state_dict
from lm.data.sft import build_sft
from experiments.d1.run import load_base, parser as d1_parser, train_sft

TARGETS = ["attn.wq", "attn.wk", "attn.wv", "attn.wo", "mlp.w1", "mlp.w2", "mlp.w3"]


def run(args, pre: tuple, sft: dict, r: int, scaling: str, lr: float, tag: str, **kw) -> dict:
    model = apply_lora(load_base(args.base), TARGETS, r=r, alpha=args.alpha, scaling=scaling, **kw)
    params = [p for p in model.parameters() if p.requires_grad]
    args.lr, args.lam = lr, 0.0
    hist = train_sft(model, params, sft, args, Path(args.out), tag, pre)
    torch.save({"r": r, "alpha": args.alpha, "scaling": scaling, "kw": kw, "adapter": lora_state_dict(model)},
               Path(args.out) / f"{tag}.pt")
    n = sum(p.numel() for p in params)
    return {"tag": tag, "r": r, "scaling": scaling, "lr": lr, "trainable": n, **kw,
            "final": {k: v for k, v in hist[-1].items() if k != "entropy_bins"},
            "curve": [(h["step"], h["val_loss"]) for h in hist]}


def plot(results: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, scaling in zip(axes, ("standard", "rs")):
        for res in (x for x in results if x["scaling"] == scaling and "dora" not in x and "init" not in x):
            s, v = zip(*res["curve"])
            ax.plot(s, v, label=f"r={res['r']}")
        ax.set_title("γ = α/r" if scaling == "standard" else "γ = α/√r")
        ax.set_xlabel("step")
        ax.legend()
    axes[0].set_ylabel("verse val loss (nats/token)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pre = tokendata_source(args.data, args.val_windows, args.T_pre, "cuda")
    sft, _, _ = build_sft(Tokenizer.load(args.tok), args.T, args.verse, plain=True)
    lr = args.lr
    if args.tune_lrs:
        tuned = [run(args, pre, sft, 8, "standard", x, f"tune_lr{x}") for x in args.tune_lrs]
        lr = min(tuned, key=lambda x: x["final"]["val_loss"])["lr"]
        print("tuned lr at r=8:", lr)
    results = [run(args, pre, sft, r, s, lr, f"{s}_r{r}") for s in ("standard", "rs") for r in args.ranks]
    results += [run(args, pre, sft, 16, args.variant_scaling, lr, "dora_r16", dora=True),
                run(args, pre, sft, 16, args.variant_scaling, lr, "pissa_r16", init="pissa")]
    (out / "sweep.json").write_text(json.dumps(results, indent=1))
    plot(results, out / "rank_sweep.png")
    for x in results:
        print(x["tag"], x["trainable"], round(x["final"]["val_loss"], 4), round(x["final"]["fineweb_bpb"], 4),
              round(x["final"].get("step_time_s", 0), 4))


if __name__ == "__main__":
    ap = d1_parser()
    ap.set_defaults(out="runs/d2/sweep", lr=2e-3, epochs=2, eval_every=10)
    ap.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--tune_lrs", type=float, nargs="*", default=[])
    ap.add_argument("--variant_scaling", default="rs", choices=["standard", "rs"])
    main(ap.parse_args())
