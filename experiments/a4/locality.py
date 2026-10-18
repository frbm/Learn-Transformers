"""A4 Task 5: E g(q, q, Delta) for RoPE, exact and Monte Carlo, against d_h (1 - ln Delta / ln b)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lm.model.rope import apply_rope, rope_angles, rope_freqs

D, BASE, N_MC = 64, 10_000.0, 4096


def main():
    deltas = np.unique(np.logspace(0, 5, 200).astype(np.int64))
    theta = rope_freqs(D, BASE).numpy()
    exact = 2 * np.cos(np.outer(deltas, theta)).sum(1)
    approx = D * (1 - np.log(deltas) / np.log(BASE))

    # Monte Carlo through apply_rope. This tests the formula and the code path together.
    cos, sin = rope_angles(D, int(deltas.max()) + 1, BASE)
    q = torch.randn(N_MC, 1, 1, D)
    q0 = apply_rope(q, cos, sin, offset=0)
    mc = np.array([(q0 * apply_rope(q, cos, sin, offset=int(d))).sum(-1).mean().item() for d in deltas])
    print(f"max |MC - exact| = {np.abs(mc - exact).max():.3f} (MC std ~ {np.sqrt(2 * D / N_MC) * 2:.3f})")
    for d in (1, 10, 100, 1000, 10_000, 100_000):
        i = np.searchsorted(deltas, d)
        print(f"Delta={d:>6}: exact {exact[i]:7.2f}  heuristic {approx[i]:7.2f}")

    plt.semilogx(deltas, exact, label="exact 2 sum cos")
    plt.semilogx(deltas, mc, ".", ms=3, label="Monte Carlo")
    plt.semilogx(deltas, approx, "--", label="d_h (1 - ln D / ln b)")
    plt.axhline(0, color="k", lw=0.5)
    plt.xlabel("Delta"); plt.ylabel("E g(q, q, Delta)"); plt.legend()
    plt.savefig("runs/a4_locality.png", dpi=150)


if __name__ == "__main__":
    main()
