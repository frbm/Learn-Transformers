"""F1 Task 7.1: DivPO pairs from the sample bank, then DPO at D3's beta. Place the result with run_f1_bank.

python -m experiments.f1.divpo --bank runs/f1/bank --system sft_T1.0 --ref-ckpt runs/d1/sft.pt --beta 0.1
python -m experiments.f1.divpo --bank runs/f1/bank_smoke --smoke
Rule "standard" (best vs worst reward) trains the control run with the same code.
"""
import argparse
import dataclasses
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lm.verse.common import header_text, load_model, load_tokenizer, read_jsonl, write_jsonl
from lm.eval.embedding import divpo_pair, embed


def build_pairs(rows: list[dict], rule: str, quantile: float) -> list[tuple[dict, dict]]:
    thr = float(np.quantile([r["reward"] for r in rows], quantile))
    by_h = defaultdict(list)
    for r in rows:
        by_h[json.dumps(r["header"], sort_keys=True)].append(r)
    pairs = []
    for g in by_h.values():
        rew = torch.tensor([r["reward"] for r in g], dtype=torch.float64)
        if rule == "divpo":
            ij = divpo_pair(embed([r["poem"] for r in g]), rew, thr)
        else:
            ij = (int(rew.argmax()), int(rew.argmin())) if rew.max() > rew.min() else None
        if ij is not None:
            pairs.append((g[ij[0]], g[ij[1]]))
    return pairs


def seq_logp(model, prompts: list[list[int]], resps: list[list[int]]) -> torch.Tensor:
    """Right padding: a causal model never sees the padding of later positions. One plain forward."""
    full = [p + r for p, r in zip(prompts, resps)]
    L, dev = max(map(len, full)), model.tok_emb.weight.device
    idx = torch.zeros(len(full), L, dtype=torch.long)
    is_resp = torch.zeros(len(full), L, dtype=torch.bool)
    for i, (p, r) in enumerate(zip(prompts, resps)):
        idx[i, : len(p) + len(r)] = torch.tensor(p + r)
        is_resp[i, len(p) : len(p) + len(r)] = True
    idx, is_resp = idx.to(dev), is_resp.to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(idx[:, :-1])
    lp = logits.float().log_softmax(-1).gather(-1, idx[:, 1:, None])[..., 0]   # fp32 log-softmax under autocast
    return (lp * is_resp[:, 1:]).sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="runs/f1/bank")
    ap.add_argument("--system", default="sft_T1.0")
    ap.add_argument("--ref-ckpt", default="runs/d1/sft.pt")
    ap.add_argument("--tok", default="data/fineweb_tok.json")
    ap.add_argument("--rule", choices=["divpo", "standard"], default="divpo")
    ap.add_argument("--quantile", type=float, default=0.5, help="reward threshold as a quantile")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="runs/f1/divpo.pt")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    tok = load_tokenizer(a.tok)
    eos = tok.special["<|endoftext|>"]
    rows = read_jsonl(Path(a.bank) / f"{a.system}.jsonl")
    pairs = build_pairs(rows, a.rule, a.quantile)
    write_jsonl(Path(a.out).with_suffix(f".{a.rule}_pairs.jsonl"), [{"chosen": c, "rejected": r} for c, r in pairs])
    print(f"{len(pairs)} {a.rule} pairs")
    ckpt = None if a.smoke else a.ref_ckpt
    policy = load_model(ckpt, tok.vocab_size, dtype=torch.float32).train()
    ref = load_model(ckpt, tok.vocab_size).eval()
    opt = torch.optim.AdamW(policy.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    enc = lambda r: (tok.encode(header_text(r["header"], r["header"]["lines"], tok)), r["ids"] + [eos] * r["finished"])
    rng = np.random.default_rng(0)
    for ep in range(a.epochs):
        accs, losses = [], []
        for b in range(0, len(pairs), a.batch):
            chunk = [pairs[i] for i in rng.permutation(len(pairs))[b : b + a.batch]]
            (pc, rc), (pr, rr) = zip(*[enc(c) for c, _ in chunk]), zip(*[enc(r) for _, r in chunk])
            with torch.no_grad():
                ref_c, ref_r = seq_logp(ref, list(pc), list(rc)), seq_logp(ref, list(pr), list(rr))
            margin = a.beta * ((seq_logp(policy, list(pc), list(rc)) - ref_c) - (seq_logp(policy, list(pr), list(rr)) - ref_r))
            loss = -F.logsigmoid(margin).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item()); accs.append((margin > 0).float().mean().item())
        print(json.dumps({"epoch": ep, "loss": float(np.mean(losses)), "pair_acc": float(np.mean(accs))}), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": dataclasses.asdict(policy.cfg), "model": policy.state_dict()},
               Path(a.out).with_suffix(".smoke.pt") if a.smoke else a.out)


if __name__ == "__main__":
    main()
