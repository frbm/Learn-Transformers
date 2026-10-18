"""D4 Task 5: "exactly n non-empty lines" by token masking vs exact conditioning (rejection sampling).

python -m experiments.d4.constrained --ckpt runs/d1/sft.pt --tok runs/d1/tok.json --n-lines 8
python -m experiments.d4.constrained --smoke
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lm.verse.common import LineConstraint, count_lines, decode, header_text, load_model, load_reward, load_tokenizer, verse_split
from lm.eval.embedding import sample_set
from lm.infer.generate import generate, rescore
from lm.infer.sampling import Sampler


def summary(rows, model, tok, reward, eos) -> dict:
    if not rows:
        return {"n": 0, "last_line_chars": [], "mean_last_line_chars": float("nan"), "logp_per_token": float("nan"),
                "reward": float("nan")}
    prompts = [tok.encode(header_text(r["header"], r["header"]["lines"], tok)) for r in rows]
    sc = rescore(model, prompts, [r["ids"] + [eos] for r in rows])
    last = [len([l for l in r["poem"].splitlines() if l.strip()][-1]) if r["poem"].strip() else 0 for r in rows]
    return {"n": len(rows), "last_line_chars": last,
            "mean_last_line_chars": float(np.mean(last)),
            "logp_per_token": (sc["logp"].sum() / sc["n"].sum()).item(),
            "reward": float(np.mean([reward(r["header"], r["poem"]) for r in rows]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tok", default="data/fineweb_tok.json")
    ap.add_argument("--reward", default=None)
    ap.add_argument("--n-lines", type=int, default=8)
    ap.add_argument("--headers", type=int, default=100)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--max-tries", type=int, default=64, help="rejection rounds of `samples` per header")
    ap.add_argument("--max-new", type=int, default=192)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="runs/d4_constrained")
    a = ap.parse_args()
    if a.smoke:
        a.headers, a.samples, a.max_tries, a.max_new, a.n_lines = 4, 8, 2, 48, 2
    tok = load_tokenizer(a.tok)
    eos = tok.special["<|endoftext|>"]
    model = load_model(a.ckpt, tok.vocab_size)
    reward = load_reward(a.reward)
    heads = [dict(m, lines=a.n_lines) for m, _ in verse_split("val", a.headers)]
    g = torch.Generator(device="cuda").manual_seed(0)
    sampler = Sampler([])
    if a.smoke:                                              # a random model rarely ends lines: favour newlines and eos
        bias = torch.tensor([6.0 * (b"\n" in v) for v in tok.vocab] + [0.0] * (model.cfg.vocab_size - tok.vocab_size),
                            device="cuda")
        bias[eos] = 4.0
        sampler = Sampler([lambda x: x + bias])

    jobs = [(h, tok.encode(header_text(h, a.n_lines, tok))) for h in heads for _ in range(a.samples)]
    con = LineConstraint(tok, model.cfg.vocab_size, torch.full((len(jobs),), a.n_lines, device="cuda"), eos)
    seqs = generate(model, [p for _, p in jobs], a.max_new, sampler, eos, generator=g, constraint=con)
    masked = [{"header": h, "ids": s, "poem": decode(tok, s)} for (h, _), s in zip(jobs, seqs) if len(s) < a.max_new]

    exact, tried = [], 0
    for h in heads:                                          # rejection: stop a header at `samples` accepted
        got = []
        for _ in range(a.max_tries):
            rows = sample_set(model, [h], a.samples, sampler, tok, a.max_new, generator=g)
            tried += len(rows)
            got += [r for r in rows if r["finished"] and count_lines(r["poem"]) == a.n_lines]
            if len(got) >= a.samples:
                break
        exact += got[: a.samples]
    res = {"n_lines": a.n_lines, "masked": summary(masked, model, tok, reward, eos),
           "rejection": summary(exact, model, tok, reward, eos),
           "acceptance_rate": len(exact) / max(1, tried), "masked_finished_frac": len(masked) / len(jobs)}
    printable = {k: ({kk: vv for kk, vv in v.items() if kk != "last_line_chars"} if isinstance(v, dict) else v)
                 for k, v in res.items()}
    print(json.dumps(printable, indent=1))
    Path(a.out).with_suffix(".json").write_text(json.dumps(res))
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in ("masked", "rejection"):
        if res[name]["last_line_chars"]:
            ax.hist(res[name]["last_line_chars"], bins=30, alpha=0.5, density=True, label=name)
    ax.set_xlabel("last line length (chars)"); ax.legend()
    fig.tight_layout()
    fig.savefig(Path(a.out).with_suffix(".png"), dpi=120)


if __name__ == "__main__":
    main()
