"""D4 Task 3: speculative decoding on real models. Exactness (chi-square), acceptance vs 1 - TV, speed-up.

python -m experiments.d4.speculative --target runs/c1/tier1.pt --draft runs/b1/tier0.pt
python -m experiments.d4.speculative --smoke      # random target, draft = target + weight noise
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy.stats import chi2_contingency

from lm.verse.common import TIER0, TIER1, load_model, load_tokenizer
from lm.infer.generate import generate
from lm.infer.sampling import Sampler
from lm.infer.speculative import expected_tokens_per_call, speculative_cached


def noisy_copy(model, sigma: float, seed: int = 1):
    """A draft that disagrees with the target by a controlled amount. Smoke runs only."""
    import copy
    d = copy.deepcopy(model)
    g = torch.Generator(device=model.tok_emb.weight.device).manual_seed(seed)
    for p in d.parameters():
        p.data.add_(torch.randn(p.shape, generator=g, device=p.device, dtype=p.dtype) * sigma * p.data.std())
    return d


def exactness(target, draft, prompt, n: int, gamma: int, sampler, seed: int) -> dict:
    """n two-token continuations each way. Chi-square on the 30 most frequent pairs plus an 'other' cell."""
    dev = target.tok_emb.weight.device
    g = torch.Generator(device=dev).manual_seed(seed)
    spec = Counter(tuple(speculative_cached(target, draft, prompt, 2, gamma, sampler, g)[0]) for _ in range(n))
    anc = Counter()
    for b in range(0, n, 1000):
        anc.update(tuple(s) for s in generate(target, [prompt] * min(1000, n - b), 2, sampler, eos_id=-1, generator=g))
    top = [k for k, _ in (spec + anc).most_common(30)]
    row = lambda c: [c[k] for k in top] + [sum(c.values()) - sum(c[k] for k in top)]
    table = [row(spec), row(anc)]
    table = [[a, b] for a, b in zip(*table) if a + b > 0]
    stat, p, dof, _ = chi2_contingency(list(zip(*table)))
    return {"chi2": stat, "dof": dof, "p_value": p, "n": n}


def time_it(fn) -> float:
    torch.cuda.synchronize()
    t = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=None)
    ap.add_argument("--draft", default=None)
    ap.add_argument("--tok", default="data/fineweb_tok.json")
    ap.add_argument("--prompt", default="The history of the city begins")
    ap.add_argument("--n-exact", type=int, default=20_000)
    ap.add_argument("--gammas", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--draft-noise", type=float, default=0.3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="runs/d4_speculative")
    a = ap.parse_args()
    if a.smoke:
        a.n_exact, a.gammas, a.tokens, a.reps = 300, [1, 4], 32, 1
    tok = load_tokenizer(a.tok)
    V = tok.vocab_size
    target = load_model(a.target, V, shape=TIER1, dtype=torch.float32)
    draft = load_model(a.draft, V, shape=TIER0, dtype=torch.float32) if a.draft else noisy_copy(target, a.draft_noise)
    prompt = tok.encode(a.prompt)
    sampler = Sampler([])
    res = {"exactness": exactness(target, draft, prompt, a.n_exact, 3, sampler, 0)}
    print(json.dumps(res["exactness"]), flush=True)

    g = torch.Generator(device="cuda").manual_seed(1)
    t_anc = sum(time_it(lambda: generate(target, [prompt], a.tokens, sampler, -1, g)) for _ in range(a.reps)) / a.reps
    t_draft = sum(time_it(lambda: generate(draft, [prompt], a.tokens, sampler, -1, g)) for _ in range(a.reps)) / a.reps
    c = t_draft / t_anc
    res["ancestral_s"], res["draft_cost_c"], res["gammas"] = t_anc, c, []
    for gamma in a.gammas:
        stats, dt = [], 0.0
        for _ in range(a.reps):
            dt += time_it(lambda: stats.append(speculative_cached(target, draft, prompt, a.tokens, gamma, sampler, g)[1]))
        acc = sum(s["accepted"] for s in stats) / max(1, sum(s["proposed"] for s in stats))
        tpc = a.tokens * len(stats) / sum(s["target_calls"] for s in stats)
        r = {"gamma": gamma, "alpha": acc, "tokens_per_call": tpc, "predicted_tokens_per_call": expected_tokens_per_call(acc, gamma),
             "speedup": t_anc / (dt / a.reps), "predicted_speedup": expected_tokens_per_call(acc, gamma) / (1 + gamma * c),
             "accept_rate_at": stats[-1]["accept_rate_at"], "one_minus_tv_at": stats[-1]["one_minus_tv_at"]}
        res["gammas"].append(r)
        print(json.dumps(r), flush=True)
    Path(a.out).with_suffix(".json").write_text(json.dumps(res, indent=1))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for r in res["gammas"]:
        ax[0].plot(r["accept_rate_at"], "-o", label=f"gamma {r['gamma']} accepted")
        ax[0].plot(r["one_minus_tv_at"], "k--", alpha=0.4)
    ax[0].set_xlabel("draft position"); ax[0].set_ylabel("rate"); ax[0].legend(fontsize=7)
    gs = [r["gamma"] for r in res["gammas"]]
    ax[1].plot(gs, [r["tokens_per_call"] for r in res["gammas"]], "-o", label="tokens / target call")
    ax[1].plot(gs, [r["predicted_tokens_per_call"] for r in res["gammas"]], "--", label="Prop. 6")
    ax[1].plot(gs, [r["speedup"] for r in res["gammas"]], "-s", label="wall-clock speed-up")
    ax[1].set_xlabel("gamma"); ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(a.out).with_suffix(".png"), dpi=120)


if __name__ == "__main__":
    main()
