"""F1 Task 7.2: three levers on one axis. Control field, residual steering, plain instruction.

For each lever and strength: target property score, KL(q_lever || p_unsteered) by Monte Carlo, within-prompt Vendi.
python -m experiments.f1.levers --ckpt runs/d1/sft.pt --tok runs/d1/tok.json --steer-vec runs/e2/elegy_vec.pt --layer 4
python -m experiments.f1.levers --smoke
"""
import argparse
import contextlib
import importlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lm.verse.common import count_lines, header_text, load_model, load_tokenizer, steer, verse_split
from lm.eval.diversity import vendi
from lm.eval.embedding import embed, sample_set
from lm.infer.generate import rescore
from lm.infer.sampling import Sampler

ELEGIAC = set("grave mourn death dead weep tears sorrow grief lost loss farewell tomb dust ashes sigh lament "
              "elegy night cold fading fade wither silent silence pale memory remember gone departed".split())


def lexicon_score(meta: dict, poem: str) -> float:
    """Stand-in property score: fraction of words in a small elegiac lexicon."""
    w = [x.strip(".,;:!?'\"").lower() for x in poem.split()]
    return sum(x in ELEGIAC for x in w) / max(1, len(w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tok", default="data/fineweb_tok.json")
    ap.add_argument("--property", default=None, help="module:function(meta, poem) -> float")
    ap.add_argument("--steer-vec", default=None)
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--control-authors", nargs="+", default=["Thomas Gray", "Alfred Tennyson"],
                    help="control-token lever: the author field set to elegists")
    ap.add_argument("--instructions", nargs="+", default=["Write in an elegiac tone", "An elegy, mournful and slow"])
    ap.add_argument("--headers", type=int, default=50)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="runs/f1/levers")
    a = ap.parse_args()
    if a.smoke:
        a.headers, a.samples, a.max_new, a.alphas, a.out = 3, 4, 24, [4.0], a.out + "_smoke"
    tok = load_tokenizer(a.tok)
    eos = tok.special["<|endoftext|>"]
    model = load_model(a.ckpt, tok.vocab_size)
    prop = lexicon_score if not a.property else getattr(importlib.import_module(a.property.split(":")[0]),
                                                        a.property.split(":")[1])
    base = [dict(m, lines=count_lines(t)) for m, t in verse_split("val", a.headers)]
    vec = torch.randn(model.cfg.d_model) if a.smoke or not a.steer_vec else torch.load(a.steer_vec)
    vec = vec / vec.norm()
    levers = [("none", 0.0, base, "", None)]
    levers += [("control", i + 1, [dict(h, author=au) for h in base], "", None) for i, au in enumerate(a.control_authors)]
    levers += [("instruction", i + 1, base, ins, None) for i, ins in enumerate(a.instructions)]
    levers += [("steer", al, base, "", al) for al in a.alphas]
    g = torch.Generator(device="cuda").manual_seed(0)
    results = []
    for name, strength, heads, ins, alpha in levers:
        ctx = steer(model, a.layer, vec, alpha) if alpha else contextlib.nullcontext()
        with ctx:
            rows = sample_set(model, heads, a.samples, Sampler([]), tok, a.max_new, generator=g, instruction=ins)
        # KL to the unsteered model on the ORIGINAL header: E_q[log q(y | lever) - log p(y | x)].
        orig = [tok.encode(header_text(h, h["lines"], tok)) for h in base for _ in range(a.samples)]
        logp0 = rescore(model, orig, [r["ids"] + [eos] * r["finished"] for r in rows])["logp"]
        kl = float(np.mean([r["logq"] for r in rows]) - logp0.mean().item())
        E = embed([r["poem"] for r in rows])
        groups = [list(range(i, i + a.samples)) for i in range(0, len(rows), a.samples)]
        res = {"lever": name, "strength": strength, "property": float(np.mean([prop(r["header"], r["poem"]) for r in rows])),
               "kl_to_unsteered": kl, "vendi": float(np.mean([vendi(E[gr] @ E[gr].T) for gr in groups]))}
        results.append(res)
        print(json.dumps(res), flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(results, indent=1))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    by = defaultdict(list)
    for r in results:
        by[r["lever"]].append(r)
    for lev, rs in by.items():
        ax[0].plot([r["kl_to_unsteered"] for r in rs], [r["property"] for r in rs], "-o", label=lev)
        ax[1].plot([r["vendi"] for r in rs], [r["property"] for r in rs], "-o", label=lev)
    ax[0].set_xlabel("KL(q || unsteered) nats per poem"); ax[1].set_xlabel("within-prompt Vendi")
    for x in ax:
        x.set_ylabel("property score"); x.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=120)


if __name__ == "__main__":
    main()
