import math

import torch

from lm.rl.bandits import run_bandits, ucb_bound
from lm.rl.estimators import gae, reinforce_grad, sample_trajectories, weighted_score
from lm.rl.tabular import occupancy, pg_exact, policy_evaluation, random_mdp, softmax_policy

f64 = dict(dtype=torch.float64)


def test_occupancy_is_a_distribution_and_matches_the_neumann_series():
    m = random_mdp(6, 3, 0.8, seed=1)
    pi = softmax_policy(torch.randn(6, 3, **f64))
    d = occupancy(m, pi)
    P_pi = torch.einsum("sa,sat->st", pi, m.P)
    series = sum((1 - m.gamma) * m.gamma ** t * m.rho @ torch.linalg.matrix_power(P_pi, t) for t in range(400))
    torch.testing.assert_close(d, series)
    assert abs(d.sum().item() - 1) < 1e-12


def test_weighted_score_matches_autograd():
    """Two code paths: the closed-form softmax score and autograd of sum_t c_t log pi(a_t | s_t)."""
    m = random_mdp(4, 3, 0.9, seed=0)
    theta = torch.randn(4, 3, **f64, requires_grad=True)
    traj = sample_trajectories(m, softmax_policy(theta.detach()), n=3, H=7, seed=1)
    coef = torch.randn(3, 7, **f64)
    g = weighted_score(theta.detach(), traj, coef)
    for i in range(3):
        logp = torch.log_softmax(theta, -1)[traj["states"][i, :-1], traj["actions"][i]]
        (gi,) = torch.autograd.grad((coef[i] * logp).sum(), theta)
        torch.testing.assert_close(g[i], gi)


def test_whole_return_and_baselines_are_unbiased():
    m = random_mdp(5, 3, 0.5, seed=0)
    theta = torch.randn(5, 3, **f64)
    traj = sample_trajectories(m, softmax_policy(theta), n=40_000, H=40, seed=2)
    V = policy_evaluation(m, softmax_policy(theta))
    for kw in (dict(reward_to_go=False), dict(baseline=V), dict(reward_to_go=False, baseline=torch.ones(5, **f64))):
        g = reinforce_grad(theta, traj, gamma=m.gamma, **kw)
        se = g.std(0) / math.sqrt(g.shape[0])
        assert ((g.mean(0) - pg_exact(m, theta)).abs() <= 5 * se + 1e-3).all()


def test_gae_with_exact_critic_is_unbiased():                       # Proposition 6
    m = random_mdp(5, 3, 0.5, seed=0)
    theta = torch.randn(5, 3, **f64)
    pi = softmax_policy(theta)
    V = policy_evaluation(m, pi)
    traj = sample_trajectories(m, pi, n=40_000, H=40, seed=3)
    disc = m.gamma ** torch.arange(40, **f64)
    for lam in (0.0, 0.7):
        adv = gae(traj["rewards"], V[traj["states"]], m.gamma, lam)
        g = weighted_score(theta, traj, disc * adv)
        se = g.std(0) / math.sqrt(g.shape[0])
        assert ((g.mean(0) - pg_exact(m, theta)).abs() <= 5 * se + 1e-6).all()


def test_ucb_regret_is_below_its_bound_and_sublinear():
    means = torch.tensor([0.9, 0.5, 0.4, 0.2], **f64)
    R = run_bandits("ucb", means, T=3000, seeds=20).mean(0)
    assert (R <= ucb_bound(means, 3000)).all()
    assert R[-1] - R[1499] < R[1499]
