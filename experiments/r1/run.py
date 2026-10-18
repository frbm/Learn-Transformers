"""R1 Tasks 2-5. Usage: python -m experiments.r1.run {pg,variance,is,bandits,all} [--smoke]. Plots and JSON go to runs/."""
import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from lm.rl.bandits import run_bandits, ucb_bound
from lm.rl.estimators import discounts, gae, reinforce_grad, sample_trajectories, weighted_score
from lm.rl.tabular import (f64, objective, occupancy, pg_exact, policy_evaluation, policy_iteration, q_values,
                        random_mdp, softmax_policy, stack_mdps, value_iteration)

RUNS = Path.cwd() / "runs"


def save(name: str, rec: dict, fig=None) -> None:
    RUNS.mkdir(exist_ok=True)
    (RUNS / f"r1_{name}.json").write_text(json.dumps(rec, indent=1))
    if fig is not None:
        fig.tight_layout()
        fig.savefig(RUNS / f"r1_{name}.png", dpi=120)
        plt.close(fig)


def band(ax, curves: torch.Tensor, label: str, x=None) -> None:
    q = torch.quantile(curves, torch.tensor([0.25, 0.5, 0.75], **f64), dim=0)
    x = torch.arange(curves.shape[1]) if x is None else x
    ax.plot(x, q[1], label=label)
    ax.fill_between(x, q[0], q[2], alpha=0.25)


# ---------------------------------------------------------------- Task 2
def ascent(m, theta, iters: int, step) -> torch.Tensor:
    """J* - J along the iterations -> (batch of MDPs, iters + 1). One batched solve per step for all MDPs."""
    J_star = (m.rho * value_iteration(m)[0]).sum(-1)
    gaps = []
    for k in range(iters + 1):
        gaps.append(J_star - objective(m, softmax_policy(theta)))
        if k < iters:
            theta = theta + step(m, theta)
    return torch.stack(gaps, -1)


def pg_step(eta):
    return lambda m, th: eta * pg_exact(m, th)


def npg_step(eta):
    # Proposition 7's closed form. npg_direction (the pinv path) gives the same policy; the test checks it.
    def step(m, th):
        pi = softmax_policy(th)
        V = policy_evaluation(m, pi)
        return eta * (q_values(m, V) - V[..., None]) / (1 - m.gamma)
    return step


def task_pg(a) -> None:
    S, A, gamma, iters = 20, 4, 0.95, a.iters
    tune = stack_mdps([random_mdp(S, A, gamma, seed=1000)])
    grid = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    final = {eta: ascent(tune, torch.zeros(1, S, A, **f64), iters, pg_step(eta))[0, -1].item() for eta in grid}
    eta_pg, eta_npg = min(final, key=final.get), a.eta_npg
    m = stack_mdps([random_mdp(S, A, gamma, seed=s) for s in range(a.mdps)])
    V_star = value_iteration(m)[0]
    J_star = (m.rho * V_star).sum(-1)
    zero = torch.zeros(a.mdps, S, A, **f64)
    trace = []
    policy_iteration(m, trace)
    pi_gap = J_star[:, None] - torch.stack(trace, -1)
    # Bad start: large theta on the worst action under Q*. The policy is near-deterministic and suboptimal.
    theta0 = a.bad_theta * torch.nn.functional.one_hot(q_values(m, V_star).argmin(-1), A).to(torch.float64)
    C = {"PG": ascent(m, zero, iters, pg_step(eta_pg)),
         "NPG": ascent(m, zero, iters, npg_step(eta_npg)),
         "PI": torch.cat([pi_gap, pi_gap[:, -1:].expand(-1, iters + 1 - pi_gap.shape[1])], 1),
         "PG plateau": ascent(m, theta0, a.plateau_iters, pg_step(eta_pg)),
         "NPG plateau": ascent(m, theta0, a.plateau_iters, npg_step(eta_npg))}
    C = {k: v.clamp_min(1e-16) for k, v in C.items()}

    def first_below(c, frac=0.1):
        hit = (c.median(0).values < frac * c[:, 0].median()).nonzero()
        return int(hit[0]) if len(hit) else None

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k in ("PG", "NPG", "PI"):
        band(ax[0], C[k], k)
    for k in ("PG plateau", "NPG plateau"):
        band(ax[1], C[k], k)
    for x, t in zip(ax, ("start theta = 0", f"start: theta = {a.bad_theta} on the worst action")):
        x.set(yscale="log", xlabel="iteration", ylabel="J* - J", title=t)
        x.legend()
    rec = {"eta_pg": eta_pg, "tuning_final_gap": final, "eta_npg": eta_npg, "iters": iters, "mdps": a.mdps,
           "median_gap_at_iter": {k: {i: C[k][:, i].median().item() for i in (0, 10, 100, 1000, iters)
                                      if i < C[k].shape[1]} for k in C},
           "median_final_gap": {k: C[k][:, -1].median().item() for k in C},
           "iters_to_10pct_gap": {k: first_below(C[k]) for k in C},
           "pi_iterations": (pi_gap > 1e-10).sum(1).add(1).tolist()}
    print(json.dumps(rec, indent=1))
    save("pg_npg_pi", rec, fig)


# ---------------------------------------------------------------- Task 3
def task_variance(a) -> None:
    S, A, gamma, H, N = 10, 3, 0.9, a.H, a.N
    m = random_mdp(S, A, gamma, seed=7)
    theta = torch.randn(S, A, generator=torch.Generator().manual_seed(7), **f64)
    pi = softmax_policy(theta)
    V = policy_evaluation(m, pi)
    exact = pg_exact(m, theta)
    print(f"truncation bias bound gamma^H / (1 - gamma) = {gamma ** H / (1 - gamma):.1e}")
    traj = sample_trajectories(m, pi, N, H, seed=1)
    pilot = sample_trajectories(m, pi, max(N // 5, 1000), H, seed=2)
    disc = discounts(gamma, H)
    zero = {**pilot, "rewards": torch.zeros_like(pilot["rewards"])}

    def opt_baseline_grad(tr):
        # Per coordinate: g_j(b) = g0_j - b_j h_j with h = sum_t gamma^t psi_t. b*_j = E[g0 h] / E[h^2].
        # Estimate b* on an independent pilot batch. On the same batch, b* correlates with g0 (bias).
        g0p = reinforce_grad(theta, pilot, gamma)
        hp = weighted_score(theta, zero, disc.expand(pilot["rewards"].shape))
        b = (g0p * hp).mean(0) / (hp ** 2).mean(0).clamp_min(1e-300)
        return reinforce_grad(theta, tr, gamma) - b * weighted_score(theta, tr, disc.expand(tr["rewards"].shape))

    const_b = torch.full((S,), (occupancy(m, pi) @ V).item(), **f64)
    est = {"whole return": reinforce_grad(theta, traj, gamma, reward_to_go=False),
           "reward-to-go": reinforce_grad(theta, traj, gamma),
           "rtg - const b": reinforce_grad(theta, traj, gamma, baseline=const_b),
           "rtg - optimal const b*": opt_baseline_grad(traj),
           "rtg - V^pi": reinforce_grad(theta, traj, gamma, baseline=V)}
    lams = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 1.0]
    gen = torch.Generator().manual_seed(3)
    eps = torch.randn(S, generator=gen, **f64)
    gae_est = {}
    for sigma in a.noise:
        V_hat = V + sigma * eps
        for lam in lams:
            adv = gae(traj["rewards"], V_hat[traj["states"]], gamma, lam)
            gae_est[(sigma, lam)] = weighted_score(theta, traj, disc * adv)

    def bias2_var(g):
        mean = g.mean(0)
        var = g.var(0).sum().item()
        # ||mean - exact||^2 holds the noise of the mean, var / N. Remove it: unbiased squared bias.
        return max(((mean - exact) ** 2).sum().item() - var / g.shape[0], 0.0), var

    ns = torch.logspace(1, math.log10(N / 10), 12, **f64)
    rows = {k: bias2_var(g) for k, g in est.items()}
    rows.update({f"GAE sigma={s} lam={l}": bias2_var(g) for (s, l), g in gae_est.items()})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k in est:
        b2, v = rows[k]
        ax[0].plot(ns, b2 + v / ns, label=k)
    for lam in (0.0, 0.5, 0.9, 0.95, 1.0):
        b2, v = rows[f"GAE sigma={a.noise[-1]} lam={lam}"]
        ax[0].plot(ns, b2 + v / ns, "--", label=f"GAE lam={lam} (sigma={a.noise[-1]})")
    ax[0].set(xscale="log", yscale="log", xlabel="n trajectories", ylabel="MSE of the gradient", title="REINFORCE vs GAE")
    ax[0].legend(fontsize=6)
    best = {}
    for sigma in a.noise:
        for n in (100, 1000):
            mse = [rows[f"GAE sigma={sigma} lam={l}"][0] + rows[f"GAE sigma={sigma} lam={l}"][1] / n for l in lams]
            ax[1].plot(lams, mse, "o-", label=f"sigma={sigma}, n={n}")
            best[f"sigma={sigma}, n={n}"] = lams[min(range(len(lams)), key=mse.__getitem__)]
    ax[1].set(yscale="log", xlabel="lambda", ylabel="MSE", title="GAE bias-variance in lambda")
    ax[1].legend(fontsize=7)
    rec = {"S": S, "A": A, "gamma": gamma, "H": H, "N": N, "noise": a.noise, "|grad J|": exact.norm().item(),
           "bias2_and_trace_var": rows, "best_lambda": best}
    print(json.dumps(rec, indent=1))
    save("variance", rec, fig)


# ---------------------------------------------------------------- Task 4
def task_is(a) -> None:
    K, N, Hmax, clip = 10, a.N_is, a.Hmax, 0.2
    b = torch.full((K,), 1 / K, **f64)
    u = torch.where(torch.arange(K) < K // 2, 1.0, -1.0).double()     # E_b[u] = 0, E_b[u^2] = 1
    g = torch.Generator().manual_seed(0)
    H = torch.arange(1, Hmax + 1, **f64)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    rec = {}
    for chi2 in (0.01, 0.1, 0.5):
        pi = b * (1 + math.sqrt(chi2) * u)                              # chi2(pi || b) = c^2 exactly
        acts = torch.multinomial(b.expand(N, -1), Hmax, replacement=True, generator=g)
        rho = pi[acts] / b[acts]
        r = (acts == 0).double()
        W = torch.cumprod(rho, 1)
        traj = W * r.cumsum(1)                                          # (i) W_H * sum_{t<H} r_t
        per_dec = (W * r).cumsum(1)                                     # (ii) sum_t (prod_{k<=t} rho_k) r_t
        clipped = (rho.clamp(max=1 + clip) * r).cumsum(1)               # (iii) sum_t min(rho_t, 1 + eps) r_t
        # Exact variance of (i): E_b[W^2 S^2] = m^H E_q[S^2], q = b rho^2 / m, S ~ Binomial(H, q_0).
        mass, q0 = 1 + chi2, (b[0] * (pi[0] / b[0]) ** 2 / (1 + chi2)).item()
        var_exact = mass ** H * (H * q0 * (1 - q0) + H ** 2 * q0 ** 2) - (H * pi[0]) ** 2
        bias_exact = H * (b[0] * min(pi[0] / b[0], 1 + clip) - pi[0])
        for est, name in ((traj, "trajectory"), (per_dec, "per-decision"), (clipped, "clipped per-token")):
            ax[0].plot(H, est.var(0), label=f"{name}, chi2={chi2}")
        ax[0].plot(H, var_exact, "k:", lw=1)
        ax[1].plot(H, clipped.mean(0) - H * pi[0], label=f"chi2={chi2} (empirical)")
        ax[1].plot(H, bias_exact, "k:", lw=1)
        rec[f"chi2={chi2}"] = {"E[W^2] at H=200 (Prop. 9)": (1 + chi2) ** Hmax,
                               "empirical E[W^2] at Hmax": (W[:, -1] ** 2).mean().item(),
                               "var trajectory, exact / empirical at Hmax": [var_exact[-1].item(), traj[:, -1].var().item()],
                               "var per-decision at Hmax": per_dec[:, -1].var().item(),
                               "var clipped at Hmax": clipped[:, -1].var().item(),
                               "bias clipped at Hmax (exact)": bias_exact[-1].item()}
    ax[0].set(yscale="log", xlabel="H", ylabel="variance", title="IS estimators (dotted: exact, trajectory)")
    ax[0].legend(fontsize=6)
    ax[1].set(xlabel="H", ylabel="bias of (iii)", title=f"clipped per-token weights, eps={clip}")
    ax[1].legend(fontsize=7)
    rec["500 tokens, chi2=1e-3: E[W^2]"] = 1.001 ** 500
    rec["500 tokens, chi2=1e-2: E[W^2]"] = 1.01 ** 500
    print(json.dumps(rec, indent=1))
    save("importance", rec, fig)


# ---------------------------------------------------------------- Task 5
def task_bandits(a) -> None:
    means = torch.rand(10, generator=torch.Generator().manual_seed(0), **f64)
    fig, ax = plt.subplots(figsize=(6, 4))
    rec = {"means": means.tolist()}
    for algo, eps, label in (("eps_greedy", 0.01, "eps-greedy 0.01"), ("eps_greedy", 0.1, "eps-greedy 0.1"),
                             ("ucb", 0.0, "UCB1"), ("thompson", 0.0, "Thompson")):
        t0 = time.perf_counter()
        R = run_bandits(algo, means, a.T, a.seeds, seed=1, eps=eps)
        band(ax, R, label, x=torch.arange(1, a.T + 1))
        rec[label] = {"final_mean_regret": R[:, -1].mean().item(), "se": (R[:, -1].std() / math.sqrt(a.seeds)).item(),
                      "seconds": time.perf_counter() - t0}
    ax.plot(torch.arange(1, a.T + 1), ucb_bound(means, a.T), "k:", label="UCB1 bound")
    ax.set(xscale="log", yscale="log", xlabel="t", ylabel="cumulative regret", title=f"K=10, {a.seeds} seeds")
    ax.legend(fontsize=7)
    rec["ucb_bound_at_T"] = ucb_bound(means, a.T)[-1].item()
    print(json.dumps(rec, indent=1))
    save("bandits", rec, fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["pg", "variance", "is", "bandits", "all"])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--plateau-iters", type=int, default=20000)
    p.add_argument("--mdps", type=int, default=20)
    p.add_argument("--eta-npg", type=float, default=1.0)
    p.add_argument("--bad-theta", type=float, default=8.0)
    p.add_argument("--H", type=int, default=120)
    p.add_argument("--N", type=int, default=100_000)
    p.add_argument("--noise", type=float, nargs="+", default=[0.3, 3.0])
    p.add_argument("--N-is", type=int, default=10_000)
    p.add_argument("--Hmax", type=int, default=200)
    p.add_argument("--T", type=int, default=10_000)
    p.add_argument("--seeds", type=int, default=100)
    a = p.parse_args()
    if a.smoke:
        a.iters, a.plateau_iters, a.mdps, a.N, a.T, a.seeds = 60, 200, 3, 5000, 500, 10
    torch.manual_seed(0)
    tasks = {"pg": task_pg, "variance": task_variance, "is": task_is, "bandits": task_bandits}
    for name in tasks if a.task == "all" else [a.task]:
        t0 = time.perf_counter()
        tasks[name](a)
        print(f"{name}: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
