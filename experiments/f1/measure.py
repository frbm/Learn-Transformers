"""F1 Tasks 3 and 4: measure every system of the bank, then the quality-diversity frontier.

python -m experiments.f1.measure --bank runs/f1/bank
python -m experiments.f1.measure --bank runs/f1/bank_smoke --smoke
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lm.verse.common import header_text, load_model, load_tokenizer, read_jsonl
from lm.eval.diversity import distinct_n, entropy_estimate, mauve, pareto_front, self_bleu, vendi
from lm.eval.embedding import embed, quantise
from lm.infer.generate import rescore


def mean_se(xs) -> tuple[float, float]:
    xs = np.asarray(xs, dtype=float)
    return float(xs.mean()), float(xs.std(ddof=1) / math.sqrt(len(xs))) if len(xs) > 1 else float("nan")


def heldout_bpb(ckpt, human: list[dict], tok, smoke: bool) -> float:
    """Proposition 4: bpb of held-out human verse under the policy, ratio of sums over all poems."""
    model = load_model(None if smoke else ckpt, tok.vocab_size)
    eos = tok.special["<|endoftext|>"]
    prompts = [tok.encode(header_text(h["header"], h["header"]["lines"], tok)) for h in human]
    sc = rescore(model, prompts, [tok.encode(h["poem"]) + [eos] for h in human], batch_size=16)
    n_bytes = sum(len(h["poem"].encode()) + 1 for h in human)
    del model
    torch.cuda.empty_cache()
    return (-sc["logp"].sum() / (math.log(2) * n_bytes)).item()


def measure(rows: list[dict], human_emb, ks) -> dict:
    by_h = defaultdict(list)
    for i, r in enumerate(rows):
        by_h[json.dumps(r["header"], sort_keys=True)].append(i)
    groups = list(by_h.values())
    E = embed([r["poem"] for r in rows])
    poems = lambda g: [rows[i]["poem"] for i in g]
    out = {"n": len(rows), "headers": len(groups)}
    out["reward"], out["reward_se"] = mean_se([np.mean([rows[i]["reward"] for i in g]) for g in groups])
    out["vendi"], out["vendi_se"] = mean_se([vendi(E[g] @ E[g].T) for g in groups])
    out["distinct2"], _ = mean_se([distinct_n(poems(g), 2) for g in groups])
    out["distinct3"], _ = mean_se([distinct_n(poems(g), 3) for g in groups])
    out["self_bleu"], _ = mean_se([self_bleu(poems(g)) for g in groups])
    if rows[0]["logq"] is not None:
        out["entropy_per_token"], out["entropy_se"] = entropy_estimate(
            torch.tensor([r["logq"] for r in rows]), torch.tensor([r["n_tokens"] for r in rows]))
    else:
        out["kl_bound"] = rows[0]["kl_bound"]
    first = E[[g[0] for g in groups]]
    out["vendi_across"] = vendi(first @ first.T)                # homogenisation: one sample per header
    for k in ks:
        hp, hq = quantise(human_emb, E, max(2, min(k, (len(rows) + len(human_emb)) // 2)))
        out[f"mauve_k{k}"] = mauve(hp, hq)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="runs/f1/bank")
    ap.add_argument("--tok", default="data/fineweb_tok.json")
    ap.add_argument("--k", type=int, default=None, help="MAUVE clusters; default #samples / 10")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="runs/f1/frontier")
    a = ap.parse_args()
    bank, tok = Path(a.bank), load_tokenizer(a.tok)
    human = read_jsonl(bank / "human.jsonl")
    human_emb = embed([h["poem"] for h in human])
    results = {}
    for f in sorted(bank.glob("*.jsonl")):
        if f.stem == "human":
            continue
        rows = read_jsonl(f)
        k = a.k or max(2, len(rows) // 10)
        res = measure(rows, human_emb, (k // 2, k, 2 * k))
        meta = json.loads((bank / f"{f.stem}.meta.json").read_text())
        if meta.get("policy"):
            res["heldout_bpb"] = heldout_bpb(meta["ckpt"], human, tok, a.smoke)
        results[f.stem] = res
        print(f.stem, json.dumps(res), flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(results, indent=1))

    names = list(results)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    pts = [(results[n]["vendi"], results[n]["reward"]) for n in names]
    for n, (x, y) in zip(names, pts):
        ax[0].errorbar(x, y, xerr=results[n]["vendi_se"], yerr=results[n]["reward_se"], fmt="o", capsize=2)
        ax[0].annotate(n + (" (BoN: KL bound)" if "kl_bound" in results[n] else ""), (x, y), fontsize=7)
    front = pareto_front(pts)
    ax[0].plot([pts[i][0] for i in front], [pts[i][1] for i in front], "k--", alpha=0.5, label="Pareto front")
    ax[0].set_xlabel("within-prompt Vendi (embedding kernel)"); ax[0].set_ylabel("mean reward"); ax[0].legend()
    pol = [n for n in names if "heldout_bpb" in results[n]]
    for n in pol:
        ax[1].scatter(-results[n]["heldout_bpb"], results[n]["reward"])
        ax[1].annotate(n, (-results[n]["heldout_bpb"], results[n]["reward"]), fontsize=7)
    ax[1].set_xlabel("- held-out human bpb (recall, higher is better)"); ax[1].set_ylabel("mean reward")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=120)


if __name__ == "__main__":
    main()
