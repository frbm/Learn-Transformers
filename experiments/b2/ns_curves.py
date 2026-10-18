"""B2 §4 Exercise 2. phi^(5) on [1e-3, 1] for the cubic and the quintic. Scalar maps, so this runs on the CPU."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CUBIC, QUINTIC = (1.5, -0.5, 0.0), (3.4445, -4.7750, 2.0315)


def phi_n(s: np.ndarray, coeffs, n: int = 5) -> np.ndarray:
    a, b, c = coeffs
    for _ in range(n):
        s = a * s + b * s**3 + c * s**5
    return s


def main():
    s = np.logspace(-3, 0, 100_001)
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, co in (("cubic", CUBIC), ("quintic", QUINTIC)):
        y = phi_n(s, co)
        above = s[np.argmax(y > 0.5)]
        # The band where the map has settled. Below 1e-2 the quintic is still growing the value.
        band = y[s >= 1e-2]
        print(f"{name:7s}: smallest s with phi^5(s) > 0.5: {above:.4g};  band on [1e-2, 1]: [{band.min():.3f}, {band.max():.3f}]")
        ax.plot(s, y, label=name)
    ax.axhline(1, color="k", lw=0.5)
    ax.axhline(0.5, color="k", lw=0.5, ls=":")
    ax.set(xscale="log", xlabel=r"$\sigma$ (after $G/\|G\|_F$)", ylabel=r"$\phi^{\circ 5}(\sigma)$", title="Five Newton-Schulz steps")
    ax.legend()
    fig.tight_layout()
    fig.savefig("runs/b2_ns_curves.png", dpi=120)


if __name__ == "__main__":
    main()
