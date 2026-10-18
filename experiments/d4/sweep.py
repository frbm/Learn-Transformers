"""D4 Task 4: the decoder sweep. Quality (reward) against diversity, one point per decoder.

python -m experiments.d4.sweep --ckpt runs/d1/sft.pt --tok runs/d1/tok.json --reward d3_reward:reward
python -m experiments.d4.sweep --smoke
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lm.verse.common import count_lines, decode, header_text, load_model, load_reward, load_tokenizer, verse_split
from lm.eval.diversity import distinct_n, self_bleu
from lm.eval.embedding import sample_set
from lm.infer.generate import beam_search, rescore
from lm.infer.sampling import make_sampler

DECODERS = ["greedy", "beam=4", *[f"temp={t}" for t in (0.6, 0.8, 1.0, 1.2, 1.4)],
            *[f"top_p={p}" for p in (0.8, 0.9, 0.95)], "min_p=0.05", "min_p=0.1", "eta=0.0003", "eta=0.001",
            "typical=0.2", "typical=0.9"]


def run_decoder(spec, model, headers, n, tok, max_new, g):
    eos = tok.special["<|endoftext|>"]
    if spec.startswith("beam="):
        rows = []
        for h in headers:
            ids = beam_search(model, tok.encode(header_text(h, h["lines"], tok)), int(spec[5:]), max_new, eos)
            rows += [{"header": h, "poem": decode(tok, ids), "ids": ids, "finished": len(ids) < max_new}] * n
        return rows
    sampler = make_sampler(spec)
    if spec == "greedy":                                     # deterministic: one sample per header, repeated
        return [r for r in sample_set(model, headers, 1, sampler, tok, max_new, generator=g) for _ in range(n)]
    return sample_set(model, headers, n, sampler, tok, max_new, generator=g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tok", default="data/fineweb_tok.json")
    ap.add_argument("--reward", default=None)
    ap.add_argument("--headers", type=int, default=200)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--decoders", nargs="+", default=DECODERS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="runs/d4_sweep")
    a = ap.parse_args()
    if a.smoke:
        a.headers, a.samples, a.max_new = 6, 4, 24
        a.decoders = ["greedy", "beam=4", "temp=0.8", "top_p=0.9", "min_p=0.1", "eta=0.001", "typical=0.9"]
    tok = load_tokenizer(a.tok)
    model = load_model(a.ckpt, tok.vocab_size)
    reward = load_reward(a.reward)
    headers = [dict(m, lines=count_lines(t)) for m, t in verse_split("val", a.headers)]
    g = torch.Generator(device="cuda").manual_seed(0)
    eos = tok.special["<|endoftext|>"]
    results = []
    for spec in a.decoders:
        rows = run_decoder(spec, model, headers, a.samples, tok, a.max_new, g)
        prompts = [tok.encode(header_text(r["header"], r["header"]["lines"], tok)) for r in rows]
        resp = [r["ids"] + [eos] * r["finished"] for r in rows]
        proc = None if spec.startswith(("beam", "greedy")) else make_sampler(spec).process
        sc = rescore(model, prompts, resp, processor=proc)
        n_tok = sc["n"].double().sum().clamp(min=1)
        by_h = [[r["poem"] for r in rows[i : i + a.samples]] for i in range(0, len(rows), a.samples)]
        res = {"decoder": spec,
               "reward": float(np.mean([reward(r["header"], r["poem"]) for r in rows])),
               "distinct2": float(np.mean([distinct_n(p, 2) for p in by_h])),
               "distinct3": float(np.mean([distinct_n(p, 3) for p in by_h])),
               "self_bleu": float(np.mean([self_bleu(p) for p in by_h])),
               "surprisal": (-sc["logp"].sum() / n_tok).item(),
               # Greedy keeps one token: the price is -log p(argmax). Beam has no per-token rule.
               "trunc_cost": float("nan") if spec.startswith("beam") else
               (-sc["logp"].sum() / n_tok).item() if spec == "greedy" else (sc["trunc"].sum() / n_tok).item()}
        results.append(res)
        print(json.dumps(res), flush=True)
    out = Path(a.out)
    out.with_suffix(".json").write_text(json.dumps(results, indent=1))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for x, key in zip(ax, ("distinct2", "self_bleu")):
        for r in results:
            x.scatter(r[key], r["reward"])
            x.annotate(r["decoder"], (r[key], r["reward"]), fontsize=7)
        x.set_xlabel(key + (" (lower = more diverse)" if key == "self_bleu" else "")); x.set_ylabel("reward")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=120)


if __name__ == "__main__":
    main()
