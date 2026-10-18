"""F1 Task 5: typicality bias in the reward. Regress reward on length-normalised log pi_ref (plus length).

python -m experiments.f1.typicality --bank runs/f1/bank --sft sft_T1.0 --dpo dpo_beta0.1 --ref-ckpt runs/d1/sft.pt --beta 0.1
python -m experiments.f1.typicality --bank runs/f1/bank_smoke --sft sft_T1.0 --dpo sft_T0.7 --smoke
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lm.verse.common import header_text, load_model, load_tokenizer, read_jsonl
from lm.eval.diversity import entropy_estimate
from lm.eval.embedding import sample_set
from lm.infer.generate import rescore
from lm.infer.sampling import make_sampler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="runs/f1/bank")
    ap.add_argument("--sft", default="sft_T1.0")
    ap.add_argument("--dpo", default="dpo_beta0.1")
    ap.add_argument("--ref-ckpt", default="runs/d1/sft.pt")
    ap.add_argument("--tok", default="data/fineweb_tok.json")
    ap.add_argument("--beta", type=float, default=0.1, help="D3's KL coefficient")
    ap.add_argument("--headers", type=int, default=50, help="headers for the tempered-SFT entropy")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.headers, a.samples, a.max_new = 4, 4, 24
    tok = load_tokenizer(a.tok)
    ref = load_model(None if a.smoke else a.ref_ckpt, tok.vocab_size)
    rows = read_jsonl(Path(a.bank) / f"{a.sft}.jsonl")
    eos = tok.special["<|endoftext|>"]
    prompts = [tok.encode(header_text(r["header"], r["header"]["lines"], tok)) for r in rows]
    # Rescore the sampled ids. Re-encoding the decoded text can give another tokenisation.
    resp = [r.get("ids", tok.encode(r["poem"])) + [eos] * r["finished"] for r in rows]
    sc = rescore(ref, prompts, resp)
    n = sc["n"].double().numpy()
    x = sc["logp"].numpy() / n                               # length-normalised log pi_ref
    y = np.array([r["reward"] for r in rows])
    X = np.stack([x, n, np.ones_like(x)], 1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    cov = resid.var(ddof=3) * np.linalg.inv(X.T @ X)
    # r ≈ a * log pi / n: per sequence, alpha ≈ a / mean(n) multiplies log pi_ref(y) (Proposition 6).
    a_tok, a_se = coef[0], float(np.sqrt(cov[0, 0]))
    alpha = a_tok / n.mean()
    T = 1 / (1 + alpha / a.beta)
    res = {"coef_per_token": a_tok, "coef_se": a_se, "coef_length": coef[1], "alpha_per_sequence": alpha,
           "beta": a.beta, "implied_T": T}
    heads = list({json.dumps(r["header"], sort_keys=True): r["header"] for r in rows}.values())[: a.headers]
    g = torch.Generator(device="cuda").manual_seed(0)
    temp = sample_set(ref, heads, a.samples, make_sampler(f"temp={max(T, 0.05)}"), tok, a.max_new, generator=g)
    H = lambda rs: entropy_estimate(torch.tensor([r["logq"] for r in rs]), torch.tensor([r["n_tokens"] for r in rs]))
    res["entropy_sft_T1"], res["entropy_sft_at_implied_T"] = H(rows), H(temp)
    dpo = Path(a.bank) / f"{a.dpo}.jsonl"
    if dpo.exists():
        res["entropy_dpo"] = H(read_jsonl(dpo))
    print(json.dumps(res, indent=1))
    Path("runs/f1").mkdir(parents=True, exist_ok=True)
    Path(f"runs/f1/typicality{'_smoke' if a.smoke else ''}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
