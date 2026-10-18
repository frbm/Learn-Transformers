"""A1 §3 exercise: reverse-KL projection of a two-Gaussian mixture onto one Gaussian.

KL(q || p) = E_q[log q - log p]. Gauss-Hermite quadrature gives E_q exactly for smooth integrands.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

NODES, WEIGHTS = np.polynomial.hermite_e.hermegauss(80)
WEIGHTS = WEIGHTS / WEIGHTS.sum()


def log_p(x, a):
    z = np.stack([-0.5 * (x + a) ** 2, -0.5 * (x - a) ** 2])
    return logsumexp(z, axis=0) - np.log(2) - 0.5 * np.log(2 * np.pi)


def reverse_kl(params, a):
    mu, log_s = params
    s = np.exp(log_s)
    x = mu + s * NODES
    neg_entropy = -0.5 * np.log(2 * np.pi * np.e) - log_s
    return neg_entropy - WEIGHTS @ log_p(x, a)


def project(a):
    # Start on both sides of 0. A single start can stay on the symmetric saddle.
    best = min(
        (minimize(reverse_kl, [mu0, 0.0], args=(a,), method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-12})
         for mu0 in (0.0, 0.5 * a + 1e-3, a)),
        key=lambda r: r.fun,
    )
    return abs(best.x[0]), np.exp(best.x[1]), best.fun


def main():
    grid = np.linspace(0, 3, 301)
    out = np.array([project(a) for a in grid])
    i = np.argmax(out[:, 0] > 1e-3)
    # A jump of |mu| at a* (not a slow growth from 0) means a first-order (discontinuous) transition.
    print(f"a* ~ {grid[i]:.3f}; |mu| jumps {out[i - 1, 0]:.3f} -> {out[i, 0]:.3f}")
    print(f"KL at a=3: {out[-1, 2]:.4f} (log 2 = {np.log(2):.4f})")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].plot(grid, out[:, 0]); ax[0].set_xlabel("a"); ax[0].set_ylabel("|mu|")
    ax[1].plot(grid, out[:, 1]); ax[1].set_xlabel("a"); ax[1].set_ylabel("sigma")
    fig.tight_layout(); fig.savefig("a1_reverse_kl.png", dpi=150)


if __name__ == "__main__":
    main()
