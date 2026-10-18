import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lm.eval.metrics import bits_per_byte
from lm.baselines.ngram import NGram


def splits(x: np.ndarray):
    # Use a contiguous split. A random split puts the same n-grams in train and test.
    n = len(x)
    return x[: int(0.8 * n)], x[int(0.8 * n) : int(0.9 * n)], x[int(0.9 * n) :]


def sweep(train, val, test, orders, alphas=(1e-3, 1e-2, 1e-1, 1.0)):
    rows = []
    for k in orders:
        m = NGram(k, alpha=1.0).fit(train)
        val_bpb = {}
        for a in alphas:
            m.alpha = a
            val_bpb[a] = bits_per_byte(m.nll(val), len(val) - k + 1)
        m.alpha = min(val_bpb, key=val_bpb.get)
        rows.append((k, m.alpha, val_bpb[m.alpha], bits_per_byte(m.nll(test), len(test) - k + 1)))
        print(f"n={k}  alpha={m.alpha:g}  val={rows[-1][2]:.3f}  test={rows[-1][3]:.3f} bpb", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus.txt")
    ap.add_argument("--fraction", type=float, default=1.0, help="0.1 for the extension")
    ap.add_argument("--out", default="a1_bpb_vs_order.png")
    args = ap.parse_args()

    x = np.frombuffer(Path(args.corpus).read_bytes(), dtype=np.uint8).astype(np.int64)
    x = x[: int(len(x) * args.fraction)]
    print(f"{len(x):,} bytes")
    rows = sweep(*splits(x), orders=range(1, 8))
    plt.plot([r[0] for r in rows], [r[3] for r in rows], marker="o")
    plt.xlabel("order n")
    plt.ylabel("test bpb")
    plt.savefig(args.out, dpi=150)


if __name__ == "__main__":
    main()
