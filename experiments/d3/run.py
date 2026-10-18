"""D3 Tasks 1-6. Usage: python -m experiments.d3.run {validate,bon,dpo,grpo,frontier} [--ckpt runs/d1/sft.pt] [--smoke].

Without --ckpt the policy is a tiny random GPT (smoke runs only). Logs, samples and plots go to runs/.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from lm.rl.policy import (EOT, autocast, decode, encode_headers, frozen_copy, grpo_update, load_headers, load_policy, sample,
                   save_json_lines, score, stats_batched)
from lm.rl.prefs import bon_kl, completion_logprobs, dpo_loss, ipo_loss, kl_estimators

RUNS = Path.cwd() / "runs"


def sync(dev) -> None:
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize()


class Ctx:
    def __init__(self, a):
        self.a = a
        self.model, self.tok = load_policy(a.ckpt, a.tok, a.device, seed=a.seed)
        self.eot = self.tok.special[EOT]
        self.dev = self.model.tok_emb.weight.device
        self.gen = torch.Generator(self.dev).manual_seed(a.seed)

    @torch.no_grad()
    def rollout(self, model, headers: list[dict], G: int) -> dict:
        """G completions per header, generated in chunks of --batch sequences. Times generation alone."""
        prompts = [p for p in encode_headers(headers, self.tok) for _ in range(G)]
        sync(self.dev)
        t0 = time.perf_counter()
        comps, lps = [], []
        for i in range(0, len(prompts), self.a.batch):
            c, l = sample(model, prompts[i : i + self.a.batch], self.a.max_new, self.eot, 1.0, self.gen, return_logp=True)
            comps, lps = comps + c, lps + l
        sync(self.dev)
        dt = time.perf_counter() - t0
        hs = [h for h in headers for _ in range(G)]
        texts = [decode(self.tok, c, self.eot) for c in comps]
        return {"headers": hs, "prompts": prompts, "comps": comps, "logp": lps, "texts": texts, "seconds": dt,
                "tokens": sum(map(len, comps))}

    def kl_to_ref(self, policy, ref, prompts, comps) -> tuple[float, float]:
        """Sequence KL(pi || ref) by sampling: sum of per-token k1 along each completion. Mean and standard error."""
        lp, m, _ = stats_batched(policy, prompts, comps, self.a.micro)
        lr, _, _ = stats_batched(ref, prompts, comps, self.a.micro)
        k = (kl_estimators(lp, lr)["k1"] * m).sum(1)
        return k.mean().item(), (k.std() / math.sqrt(len(k))).item()


def plot(path: Path, draw) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    draw(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- Task 1
def task_validate(c: Ctx) -> None:
    a = c.a
    samples_path, scores_path = RUNS / "d3_validate_samples.json", RUNS / "d3_validate_scores.json"
    if a.ranking:
        from scipy.stats import spearmanr
        ranking = json.loads(Path(a.ranking).read_text())               # sample ids, best first
        scores = json.loads(scores_path.read_text())
        human = {sid: -i for i, sid in enumerate(ranking)}
        ids = [s for s in ranking if s in scores]
        res = {k: spearmanr([human[i] for i in ids], [scores[i][k] for i in ids]).statistic
               for k in scores[ids[0]]}
        print(json.dumps(res, indent=1))
        (RUNS / "d3_validate_spearman.json").write_text(json.dumps(res, indent=1))
        return
    headers = load_headers("val", a.n_validate, a.seed)
    ro = c.rollout(c.model, headers, 1)
    _, comp = score(ro["headers"], ro["texts"])
    order = list(range(len(headers)))
    random.Random(a.seed).shuffle(order)
    # Write the samples without scores. Rank them blind, then rerun with --ranking.
    samples = [{"id": f"s{i}", "header": ro["headers"][j], "poem": ro["texts"][j]} for i, j in enumerate(order)]
    scores = {f"s{i}": {k: v[j].item() for k, v in comp.items()} for i, j in enumerate(order)}
    samples_path.write_text(json.dumps(samples, indent=1), encoding="utf-8")
    scores_path.write_text(json.dumps(scores, indent=1))
    top = sorted(scores, key=lambda s: -scores[s]["total"])[:5]
    print(f"wrote {samples_path}. Rank them, then: run_d3.py validate --ranking ranking.json")
    for s in top:
        print(f"--- {s} total={scores[s]['total']:.3f}\n{samples[int(s[1:])]['poem'][:400]}")


# ---------------------------------------------------------------- Task 2
def task_bon(c: Ctx) -> None:
    a = c.a
    headers = load_headers(a.split, a.n_headers, a.seed)
    ro = c.rollout(c.model, headers, a.n)
    R, comp = score(ro["headers"], ro["texts"])
    R = R.view(len(headers), a.n)
    ns = [n for n in (1, 2, 4, 8, 16, 32) if n <= a.n]
    pts = [{"method": "best-of-n", "n": n, "kl": bon_kl(n), "reward": R[:, :n].max(1).values.mean().item(),
            "se": (R[:, :n].max(1).values.std() / math.sqrt(len(headers))).item()} for n in ns]
    rec = {"split": a.split, "headers": len(headers), "n": a.n, "points": pts,
           "gen_tokens": ro["tokens"], "gen_seconds": ro["seconds"], "tokens_per_s": ro["tokens"] / ro["seconds"],
           "mean_len": ro["tokens"] / len(ro["comps"]), "components": {k: v.mean().item() for k, v in comp.items()}}
    print(json.dumps(rec, indent=1))
    (RUNS / f"d3_bon_{a.split}.json").write_text(json.dumps(rec, indent=1))
    torch.save({"headers": headers, "prompts": ro["prompts"], "comps": ro["comps"], "texts": ro["texts"],
                "rewards": R, "components": comp}, RUNS / f"d3_samples_{a.split}.pt")
    plot(RUNS / f"d3_bon_{a.split}.png", lambda ax: (
        ax.errorbar([p["kl"] for p in pts], [p["reward"] for p in pts], [p["se"] for p in pts], marker="o"),
        ax.set(xlabel="log n - (n - 1) / n  (KL bound, nats)", ylabel="mean best reward", title="best-of-n")))


# ---------------------------------------------------------------- Task 3
def eval_point(c: Ctx, policy, ref, headers) -> dict:
    ro = c.rollout(policy, headers, c.a.eval_G)
    R, comp = score(ro["headers"], ro["texts"])
    kl, kl_se = c.kl_to_ref(policy, ref, ro["prompts"], ro["comps"])
    return {"reward": R.mean().item(), "reward_se": (R.std() / math.sqrt(len(R))).item(), "kl": kl, "kl_se": kl_se,
            "len": ro["tokens"] / len(R), **{f"r_{k}": v.mean().item() for k, v in comp.items() if k != "total"},
            "sample": ro["texts"][0][:300]}


def task_dpo(c: Ctx) -> None:
    a = c.a
    data = torch.load(RUNS / "d3_samples_train.pt", weights_only=False)
    R, G = data["rewards"], data["rewards"].shape[1]
    best, worst = R.argmax(1), R.argmin(1)
    keep = ((R.max(1).values - R.min(1).values) > a.gap).nonzero()[:, 0].tolist()
    P = [data["prompts"][i * G] for i in keep]
    W = [data["comps"][i * G + best[i]] for i in keep]
    L = [data["comps"][i * G + worst[i]] for i in keep]
    print(f"{len(keep)} pairs of {len(R)} headers (gap > {a.gap}); mean length chosen "
          f"{sum(map(len, W)) / len(W):.1f}, rejected {sum(map(len, L)) / len(L):.1f}")
    ref = frozen_copy(c.model)
    # Cache the reference once. Training then needs only the policy.
    with torch.no_grad(), autocast(c.dev):
        ref_w = torch.cat([completion_logprobs(ref, P[i : i + a.micro], W[i : i + a.micro]) for i in range(0, len(P), a.micro)])
        ref_l = torch.cat([completion_logprobs(ref, P[i : i + a.micro], L[i : i + a.micro]) for i in range(0, len(P), a.micro)])
    val = load_headers("val", a.n_eval, a.seed + 1)
    for beta in a.betas:
        policy = frozen_copy(c.model).requires_grad_(True)
        opt = torch.optim.AdamW(policy.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
        rng = torch.Generator().manual_seed(a.seed)
        log = []
        for step in range(a.steps + 1):
            if step % a.eval_every == 0 or step == a.steps:
                rec = {"step": step, "eval": True, "method": f"{a.loss.upper()} beta={beta}", **eval_point(c, policy, ref, val)}
                print(json.dumps({k: v for k, v in rec.items() if k != "sample"}), flush=True)
                log.append(rec)
            if step == a.steps:
                break
            idx = torch.randint(len(P), (a.pairs,), generator=rng).tolist()
            with autocast(c.dev):
                pw = completion_logprobs(policy, [P[i] for i in idx], [W[i] for i in idx])
                pl = completion_logprobs(policy, [P[i] for i in idx], [L[i] for i in idx])
            rw, rl = ref_w[idx], ref_l[idx]
            loss, m = dpo_loss(pw, pl, rw, rl, beta)
            if a.loss == "ipo":
                loss = ipo_loss(pw, pl, rw, rl, beta)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0).item()
            opt.step()
            opt.zero_grad(set_to_none=True)
            # Log the chosen and rejected log-probabilities separately: both falling is likelihood displacement.
            log.append({"step": step, "loss": loss.item(), **{k: v.item() for k, v in m.items()},
                        "logp_chosen": pw.mean().item(), "logp_rejected": pl.mean().item(), "grad_norm": gn})
        save_json_lines(RUNS / f"d3_{a.loss}_beta{beta}{a.tag}.jsonl", log)
        tr = [r for r in log if not r.get("eval")]
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        ax[0].plot([r["logp_chosen"] for r in tr], label="log pi(chosen)")
        ax[0].plot([r["logp_rejected"] for r in tr], label="log pi(rejected)")
        ax[1].plot([r["margin"] for r in tr], label="margin")
        for x in ax:
            x.set(xlabel="step")
            x.legend()
        fig.tight_layout()
        fig.savefig(RUNS / f"d3_{a.loss}_beta{beta}{a.tag}.png", dpi=120)
        plt.close(fig)


# ---------------------------------------------------------------- Task 4 and 5
VARIANTS = {"grpo": dict(aggregation="sequence", scale="std", beta=0.04, eps_high=0.2, dynamic=False),
            "drgrpo": dict(aggregation="constant", scale="none", beta=0.04, eps_high=0.2, dynamic=False),
            "dapo": dict(aggregation="token", scale="std", beta=0.0, eps_high=0.28, dynamic=True)}


def task_grpo(c: Ctx) -> None:
    a = c.a
    v = dict(VARIANTS[a.variant])
    if a.beta is not None:
        v["beta"] = a.beta
    policy = c.model.requires_grad_(True)
    ref = frozen_copy(policy)
    opt = torch.optim.AdamW(policy.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    pool = load_headers("train", 4096, a.seed)
    rng = random.Random(a.seed)
    name = f"d3_{a.variant}_{a.reward}_beta{v['beta']}{a.tag}"
    log, shown = [], []
    for step in range(a.steps):
        t_gen, groups, n_zero, n_seen = 0.0, [], 0, 0
        # Dynamic sampling (DAPO): drop groups with equal rewards, then sample more prompts, up to 3 rounds.
        for _ in range(3 if v["dynamic"] else 1):
            ro = c.rollout(policy, rng.sample(pool, a.P), a.G)
            t_gen += ro["seconds"]
            Rr, comp = score(ro["headers"], ro["texts"], a.reward)
            _, verse = score(ro["headers"], ro["texts"]) if a.reward != "verse" else (None, comp)
            R = Rr.view(a.P, a.G)
            zero = R.std(1) == 0
            n_zero, n_seen = n_zero + int(zero.sum()), n_seen + a.P
            for g in range(a.P):
                if not (v["dynamic"] and zero[g]):
                    sl = slice(g * a.G, (g + 1) * a.G)
                    groups.append((R[g], ro["prompts"][sl], ro["comps"][sl], ro["texts"][sl],
                                   {k: x[sl] for k, x in verse.items()}, ro["logp"][sl]))
            if len(groups) >= a.P:
                break
        groups = groups[: a.P] or [(R[0], ro["prompts"][: a.G], ro["comps"][: a.G], ro["texts"][: a.G], {}, ro["logp"][: a.G])]
        R = torch.stack([g[0] for g in groups])
        comps = [x for g in groups for x in g[2]]
        sync(c.dev)
        t0 = time.perf_counter()
        st = grpo_update(policy, ref, opt, [p for g in groups for p in g[1]], comps, R, v["aggregation"], v["scale"],
                         v["beta"], v["eps_high"], a.mu, a.micro, a.max_new, [l for g in groups for l in g[5]])
        sync(c.dev)
        comp = {k: torch.cat([g[4][k] for g in groups]).mean().item() for k in groups[0][4]}
        rec = {"step": step, "method": f"{a.variant} ({a.reward})", "reward": R.mean().item(), "reward_max": R.max().item(),
               **{f"r_{k}": x for k, x in comp.items()}, **st,
               "truncated": sum(len(x) == a.max_new and x[-1] != c.eot for x in comps) / len(comps),
               "zero_var_frac": n_zero / n_seen, "t_gen": t_gen, "t_train": time.perf_counter() - t0}
        print(json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in rec.items()}), flush=True)
        log.append(rec)
        if step % a.show_every == 0 or step == a.steps - 1:
            shown.append({"step": step, "samples": [g[3][0] for g in groups[:3]]})
            print(f"--- step {step} sample:\n{groups[0][3][0][:300]}")
    save_json_lines(RUNS / f"{name}.jsonl", log)
    (RUNS / f"{name}_samples.json").write_text(json.dumps(shown, indent=1), encoding="utf-8")
    keys = ["reward", "kl", "entropy", "len", "zero_var_frac", "r_repetition"]
    fig, ax = plt.subplots(2, 3, figsize=(12, 6))
    for x, k in zip(ax.flat, keys):
        x.plot([r.get(k, float("nan")) for r in log])
        x.set(title=k, xlabel="step")
    fig.tight_layout()
    fig.savefig(RUNS / f"{name}.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- Task 6
def task_frontier(c: Ctx) -> None:
    a = c.a
    series = {}
    bon = RUNS / "d3_bon_val.json"
    if bon.exists():
        series["best-of-n"] = [(p["kl"], p["reward"]) for p in json.loads(bon.read_text())["points"]]
    for f in sorted(RUNS.glob("d3_*.jsonl")):
        rows = [json.loads(l) for l in f.read_text().splitlines()]
        pts = [(r["kl"], r["reward"]) for r in rows if "kl" in r and "reward" in r and r.get("eval", True)]
        if pts:
            series[f.stem[3:]] = pts[:: max(1, len(pts) // 25)]
    held = json.loads(Path(a.heldout).read_text()) if a.heldout else None

    def draw(ax, s):
        for k, pts in s.items():
            ax.plot([max(p[0], 1e-3) for p in pts], [p[1] for p in pts], "o-" if k == "best-of-n" else ".-", label=k)
        ax.set(xscale="log", xlabel="KL(pi || pi_ref) (nats)", ylabel="reward")
        ax.legend(fontsize=6)

    plot(RUNS / "d3_frontier.png", lambda ax: draw(ax, series))
    if held:
        # held: {series name: [[kl, held-out score], ...]}, from your own ranking or a judge.
        plot(RUNS / "d3_frontier_heldout.png", lambda ax: draw(ax, held))
    print(json.dumps({k: v[-1] for k, v in series.items()}, indent=1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["validate", "bon", "dpo", "grpo", "frontier"])
    p.add_argument("--ckpt", default=None, help="D1 SFT checkpoint. Default: tiny random model (smoke)")
    p.add_argument("--tok", default=str(RUNS.parent / "data/fineweb_tok.json"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new", type=int, default=128)
    p.add_argument("--batch", type=int, default=256, help="sequences per generation call")
    p.add_argument("--micro", type=int, default=32, help="sequences per forward/backward")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--tag", default="")
    p.add_argument("--n-validate", type=int, default=30)
    p.add_argument("--ranking", default=None)
    p.add_argument("--split", default="val")
    p.add_argument("--n-headers", type=int, default=200)
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--loss", choices=["dpo", "ipo"], default="dpo")
    p.add_argument("--betas", type=float, nargs="+", default=[0.03, 0.1, 0.3, 1.0])
    p.add_argument("--gap", type=float, default=0.1)
    p.add_argument("--pairs", type=int, default=16)
    p.add_argument("--n-eval", type=int, default=50)
    p.add_argument("--eval-G", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--variant", choices=list(VARIANTS), default="grpo")
    p.add_argument("--reward", choices=["verse", "naive"], default="verse")
    p.add_argument("--beta", type=float, default=None, help="override the variant's KL coefficient")
    p.add_argument("--P", type=int, default=16)
    p.add_argument("--G", type=int, default=8)
    p.add_argument("--mu", type=int, default=1)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--show-every", type=int, default=20)
    p.add_argument("--heldout", default=None)
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if a.smoke:
        a.n_validate, a.n_headers, a.n, a.max_new, a.steps, a.n_eval, a.eval_every = 6, 8, 8, 24, 3, 4, 2
        a.P, a.G, a.betas, a.show_every = 4, 4, a.betas[:1], 1
    torch.manual_seed(a.seed)
    RUNS.mkdir(exist_ok=True)
    c = Ctx(a)
    {"validate": task_validate, "bon": task_bon, "dpo": task_dpo, "grpo": task_grpo, "frontier": task_frontier}[a.task](c)


if __name__ == "__main__":
    main()
