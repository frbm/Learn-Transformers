"""F1 Task 6: is the <|lines|> control used? Exact MI per poem and per token position, then an ablation.

python -m experiments.f1.control_mi --ckpt runs/d1/sft.pt --tok runs/d1/tok.json
python -m experiments.f1.control_mi --smoke
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lm.verse.common import count_lines, decode, header_text, load_model, load_tokenizer, verse_split
from lm.eval.diversity import mi_estimate, mi_standard_error, tv
from lm.infer.generate import generate, rescore
from lm.infer.sampling import Sampler

VALUES = [4, 8, 14, 20]


def line_hist(poems: list[str], max_lines: int = 40) -> torch.Tensor:
    h = torch.bincount(torch.tensor([min(count_lines(p), max_lines) for p in poems]), minlength=max_lines + 1).double()
    return h / h.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tok", default="data/fineweb_tok.json")
    ap.add_argument("--n", type=int, default=800, help="poems, the control drawn uniformly")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="runs/f1/control_mi")
    a = ap.parse_args()
    if a.smoke:
        a.n, a.max_new, a.out = 32, 24, a.out + "_smoke"
    tok = load_tokenizer(a.tok)
    eos = tok.special["<|endoftext|>"]
    model = load_model(a.ckpt, tok.vocab_size)
    metas = [m for m, _ in verse_split("val", a.n)]
    g = torch.Generator().manual_seed(0)
    cond = torch.randint(len(VALUES), (len(metas),), generator=g)
    gg = torch.Generator(device="cuda").manual_seed(0)
    prompt = lambda m, c: tok.encode(header_text(m, VALUES[c], tok))
    seqs = generate(model, [prompt(m, int(c)) for m, c in zip(metas, cond)], a.max_new, Sampler([]), eos, generator=gg)
    resp = [s + [eos] * (len(s) < a.max_new) for s in seqs]
    # One forward pass per control value: log q(y_t | c, y_<t) for every c and every position t.
    per_c = [rescore(model, [prompt(m, c) for m in metas], resp, per_token=True)["tok_logp"] for c in range(len(VALUES))]
    logq = torch.stack([torch.stack([t.sum() for t in pc]) for pc in per_c], 1)    # (n, C)
    log_prior = torch.full((len(VALUES),), -math.log(len(VALUES)), dtype=torch.float64)
    res = {"mi_nats": mi_estimate(logq, cond, log_prior), "mi_se": mi_standard_error(logq, cond, log_prior),
           "H_C": math.log(len(VALUES))}
    # Per position: the increment of the pointwise MI of the prefix y_<=t. Its sum over t is the poem's MI.
    rel_bins, nl_terms, other_terms = np.zeros(20), [], []
    rel_counts = np.zeros(20)
    for i, s in enumerate(resp):
        cum = torch.stack([pc[i].cumsum(0) for pc in per_c], 1)                    # (n_i, C)
        m = cum[:, cond[i]] - torch.logsumexp(cum + log_prior, 1)
        inc = torch.diff(m, prepend=m.new_zeros(1)).numpy()
        for t, (v, d) in enumerate(zip(s, inc)):
            b = min(19, int(20 * t / len(s)))
            rel_bins[b] += d; rel_counts[b] += 1
            (nl_terms if v == eos or b"\n" in tok.vocab[v] else other_terms).append(d)
    res["mi_per_token_at_newline_or_eos"] = float(np.mean(nl_terms)) if nl_terms else float("nan")
    res["mi_per_token_elsewhere"] = float(np.mean(other_terms)) if other_terms else float("nan")
    poems = [decode(tok, s) for s in seqs]
    res["lines_accuracy_pm1"] = float(np.mean([abs(count_lines(p) - VALUES[c]) <= 1 for p, c in zip(poems, cond)]))
    # Ablation: random control values at generation. TV between the line-count laws, per requested value.
    rand = torch.randint(len(VALUES), (len(metas),), generator=g)
    abl = generate(model, [prompt(m, int(c)) for m, c in zip(metas, rand)], a.max_new, Sampler([]), eos, generator=gg)
    abl_poems = [decode(tok, s) for s in abl]
    res["ablation_tv_by_value"] = {}
    for c, v in enumerate(VALUES):
        sel = (cond == c).nonzero()[:, 0].tolist()           # the same items, intended value c, random value given
        if sel:
            res["ablation_tv_by_value"][v] = tv(line_hist([poems[i] for i in sel]), line_hist([abl_poems[i] for i in sel]))
    print(json.dumps(res, indent=1))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(np.arange(20) / 20 + 0.025, rel_bins / np.maximum(rel_counts, 1), width=0.045)
    ax.set_xlabel("relative position in the poem"); ax.set_ylabel("mean MI increment per token (nats)")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=120)


if __name__ == "__main__":
    main()
