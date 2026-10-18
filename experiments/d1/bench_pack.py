"""D1 Task 2.3. One epoch of SFT data three ways: padded, packed with a dense mask, packed with FlexAttention.

Throughput counts useful (non-padding) tokens per second of forward + backward.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from lm.tokenizer.bpe import Tokenizer
from lm.verse.corpus import load_verse
from experiments.d1.run import load_base
from lm.data.sft import SPECIALS, add_special_tokens, format_example, pack, sft_loss


def padded(examples, pad_id: int) -> dict:
    """One example per row, padded to the longest example."""
    L = max(len(e.prompt) + len(e.response) - 1 for e in examples)
    R = len(examples)
    ids = torch.full((R, L), pad_id)
    lab = torch.full((R, L), -100)
    doc = torch.full((R, L), -1)
    for i, e in enumerate(examples):
        full = torch.tensor(e.prompt + e.response)
        n = len(full) - 1
        ids[i, :n], lab[i, :n], doc[i, :n] = full[:-1], full[1:], i
        lab[i, : len(e.prompt) - 1] = -100
    return {"input_ids": ids, "labels": lab, "doc_ids": doc}


def bench(model, b: dict, mode: str, B: int, max_batches: int) -> dict:
    n_rows = len(b["input_ids"]) // B * B
    starts = list(range(0, n_rows, B))[:max_batches]

    def step(i):
        g = {k: v[i : i + B].cuda() for k, v in b.items() if v.dim() == 2}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if mode == "padded":
                logits = model(g["input_ids"])          # causal over the row, padding at the end
            else:
                logits = model(g["input_ids"], doc_ids=g["doc_ids"], position_ids=g["position_ids"],
                               flex=mode == "flex")
        sft_loss(logits, g["labels"], g["doc_ids"]).backward()
        return int((g["doc_ids"] >= 0).sum())

    step(starts[0])                                     # warm-up: compile and allocate
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0, tok = time.perf_counter(), 0
    for i in starts:
        tok += step(i)
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {"mode": mode, "rows": n_rows, "batches": len(starts), "useful_tok_per_s": tok / dt,
            "peak_gib": torch.cuda.max_memory_allocated() / 2**30}


def main(args):
    tok = Tokenizer.load(args.tok)
    model = load_base(args.base)
    add_special_tokens(model, tok, SPECIALS)
    exs = [format_example(m, t, tok) for m, t in load_verse(args.verse)["train"]]
    exs = [e for e in exs if len(e.prompt) + len(e.response) - 1 <= args.T]
    eot = tok.special["<|endoftext|>"]
    pk = pack(exs, args.T, eot)
    pd = padded(exs, eot)
    lens = np.array([len(e.prompt) + len(e.response) - 1 for e in exs])
    pred = {"padded_useful_frac": float(lens.mean() / pd["input_ids"].shape[1]),
            "packed_useful_frac": float((pk["doc_ids"] >= 0).float().mean()),
            "flex_attn_work_frac": float((lens**2).sum() / (len(pk["input_ids"]) * args.T**2))}
    print("prediction inputs:", pred)
    # Same number of useful tokens per batch on average: scale the padded batch by the row ratio.
    B_pad = max(1, round(args.B * len(pd["input_ids"]) / len(pk["input_ids"])))
    res = [bench(model, pd, "padded", B_pad, args.batches * len(pd["input_ids"]) // len(pk["input_ids"])),
           bench(model, pk, "dense", args.B, args.batches), bench(model, pk, "flex", args.B, args.batches)]
    for r in res:
        print(r)
    Path(args.out).write_text(json.dumps({"prediction_inputs": pred, "results": res}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="runs/c1_tier1/model.pt")
    ap.add_argument("--tok", default="data/fineweb_tok.json")
    ap.add_argument("--verse", default="data/verse.jsonl")
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--batches", type=int, default=10_000, help="cap on packed batches; the default is one epoch")
    ap.add_argument("--out", default="runs/d1/bench_pack.json")
    main(ap.parse_args())
