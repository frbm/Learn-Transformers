"""C2 Task 3: the empirical LSH S-curve against Proposition 3, for random-word documents perturbed in a controlled way.

python -m experiments.c2.scurve [--b 13 --r 9 --pairs 4000]
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lm.data.corpus import MinHasher, jaccard, lsh_candidate_prob, shingles

WORDS = [f"w{i}" for i in range(20_000)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=13)
    ap.add_argument("--r", type=int, default=9)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--pairs", type=int, default=4000)
    ap.add_argument("--words", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    mh = MinHasher(num_perm=args.b * args.r, seed=args.seed + 1)
    J, cand, est = np.empty(args.pairs), np.empty(args.pairs, bool), np.empty(args.pairs)
    for i in range(args.pairs):
        a = rng.choice(WORDS, args.words)
        b = a.copy()
        # A uniform fraction of replaced words spreads the pairs over the whole J axis.
        idx = rng.choice(args.words, int(rng.uniform(0, 0.6) * args.words), replace=False)
        b[idx] = rng.choice(WORDS, len(idx))
        sa, sb = shingles(" ".join(a), args.k), shingles(" ".join(b), args.k)
        J[i] = jaccard(sa, sb)
        siga, sigb = mh.signature(sa), mh.signature(sb)
        eq = (siga == sigb).reshape(args.b, args.r)
        cand[i] = eq.all(1).any()
        est[i] = eq.mean()
    bins = np.linspace(0, 1, 21)
    which = np.digitize(J, bins) - 1
    rows = []
    for j in range(20):
        m = which == j
        if m.sum() >= 20:
            c = (bins[j] + bins[j + 1]) / 2
            p, n = cand[m].mean(), m.sum()
            # Average the curve over the pairs, not the curve at the mean J: on the steep part, f(E J) != E f(J).
            pred = float(np.mean([lsh_candidate_prob(x, args.b, args.r) for x in J[m]]))
            rows.append({"J": c, "n": int(n), "empirical": p, "predicted": pred,
                         "se": float(np.sqrt(max(pred * (1 - pred), 1 / n) / n))})
    # The standard error under the null (the prediction is right). The empirical p is 0 or 1 in many bins,
    # and its own binomial error is then 0: any difference would look infinitely significant.
    gap = max(abs(r["empirical"] - r["predicted"]) / r["se"] for r in rows)
    print(f"{'J':>5} {'n':>5} {'empirical':>10} {'predicted':>10}")
    for r in rows:
        print(f"{r['J']:5.2f} {r['n']:5d} {r['empirical']:10.3f} {r['predicted']:10.3f}")
    print(f"MinHash estimate: mean |J_hat - J| = {np.abs(est - J).mean():.4f}, bias = {(est - J).mean():+.4f}; "
          f"largest S-curve gap = {gap:.1f} standard errors")
    json.dump(rows, open("runs/c2_scurve.json", "w"), indent=1)
    js = np.linspace(0, 1, 200)
    plt.plot(js, [lsh_candidate_prob(x, args.b, args.r) for x in js], label=f"1 - (1 - J^{args.r})^{args.b}")
    plt.errorbar([r["J"] for r in rows], [r["empirical"] for r in rows], yerr=[2 * r["se"] for r in rows], fmt="o",
                 ms=3, label="empirical (2 se)")
    plt.xlabel("exact Jaccard J"); plt.ylabel("P(candidate)"); plt.legend()
    plt.savefig("runs/c2_scurve.png", dpi=150)


if __name__ == "__main__":
    main()
