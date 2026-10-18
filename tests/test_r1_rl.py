# tests/test_r1_rl.py
import dataclasses, itertools, math, pytest, torch
from lm.rl.tabular import (random_mdp, policy_evaluation, q_values, occupancy, objective, value_iteration,
                        policy_iteration, softmax_policy, pg_exact, npg_direction)
from lm.rl.estimators import sample_trajectories, reinforce_grad, gae, is_second_moment
from lm.rl.soft import h_function, tilted_policy

f64 = dict(dtype=torch.float64)

def rand_pi(S, A, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.softmax(torch.randn(S, A, generator=g, **f64), -1)

def test_policy_evaluation_is_the_bellman_fixed_point():          # Propositions 1-2
    m, pi = random_mdp(5, 3, 0.9, seed=0), rand_pi(5, 3, 1)
    V = policy_evaluation(m, pi)
    torch.testing.assert_close((pi * (m.r + m.gamma * m.P @ V)).sum(-1), V)

def test_value_and_policy_iteration_agree_and_are_optimal():
    m = random_mdp(5, 3, 0.9, seed=0)
    V_star, pi_vi = value_iteration(m, tol=1e-12)
    pi_pi, _ = policy_iteration(m)
    torch.testing.assert_close(policy_evaluation(m, pi_pi), V_star)
    torch.testing.assert_close(policy_evaluation(m, pi_vi), V_star)
    for s in range(20):
        assert (policy_evaluation(m, rand_pi(5, 3, s)) <= V_star + 1e-9).all()

def test_performance_difference_lemma():                            # Proposition 3
    m, pi, pi2 = random_mdp(5, 3, 0.9, seed=0), rand_pi(5, 3, 1), rand_pi(5, 3, 2)
    V = policy_evaluation(m, pi)
    A = q_values(m, V) - V[:, None]
    rhs = (occupancy(m, pi2) * (pi2 * A).sum(-1)).sum() / (1 - m.gamma)
    torch.testing.assert_close(objective(m, pi2) - objective(m, pi), rhs)

def test_policy_gradient_theorem_matches_autograd():                 # Proposition 4
    m = random_mdp(5, 3, 0.9, seed=0)
    theta = torch.randn(5, 3, **f64, requires_grad=True)
    objective(m, softmax_policy(theta)).backward()
    torch.testing.assert_close(pg_exact(m, theta.detach()), theta.grad)

def test_npg_on_softmax_is_multiplicative_weights():                 # Proposition 7
    m, theta, eta = random_mdp(5, 3, 0.9, seed=0), torch.randn(5, 3, **f64), 0.3
    pi = softmax_policy(theta)
    V = policy_evaluation(m, pi)
    A = q_values(m, V) - V[:, None]
    torch.testing.assert_close(softmax_policy(theta + eta * npg_direction(m, theta)),
                               softmax_policy(theta + eta * A / (1 - m.gamma)))

def test_reinforce_is_unbiased():
    m = random_mdp(5, 3, 0.5, seed=0)
    theta = torch.randn(5, 3, **f64)
    traj = sample_trajectories(m, softmax_policy(theta), n=40_000, H=40, seed=0)
    g = reinforce_grad(theta, traj, gamma=m.gamma, reward_to_go=True)
    se = g.std(0) / math.sqrt(g.shape[0])
    assert ((g.mean(0) - pg_exact(m, theta)).abs() <= 5 * se + 1e-3).all()

def test_gae_limits():                                                # §3
    g = torch.Generator().manual_seed(0)
    r, V, gamma = torch.rand(4, 6, generator=g, **f64), torch.rand(4, 7, generator=g, **f64), 0.9
    torch.testing.assert_close(gae(r, V, gamma, lam=0.0), r + gamma * V[:, 1:] - V[:, :-1])
    disc = torch.tensor([gamma ** k for k in range(6)], **f64)
    mc = torch.stack([(r[:, t:] * disc[:6 - t]).sum(1) + gamma ** (6 - t) * V[:, 6] for t in range(6)], 1)
    torch.testing.assert_close(gae(r, V, gamma, lam=1.0), mc - V[:, :-1])

def test_h_transform_factorises_the_sequence_tilt():                  # Proposition 8
    g = torch.Generator().manual_seed(0)
    V, H, beta = 3, 3, 0.7
    prefixes = [p for n in range(H) for p in itertools.product(range(V), repeat=n)]
    ref = {p: torch.softmax(torch.randn(V, generator=g, **f64), 0) for p in prefixes}
    seqs = list(itertools.product(range(V), repeat=H))
    reward = {y: torch.randn((), generator=g, **f64).item() for y in seqs}
    h = h_function(ref, reward, beta)
    pol = tilted_policy(ref, h)
    prob = lambda policy, y: math.prod(policy[y[:t]][y[t]].item() for t in range(H))
    Z = sum(prob(ref, y) * math.exp(reward[y] / beta) for y in seqs)
    assert h[()] == pytest.approx(Z, rel=1e-12)
    for y in seqs:
        assert prob(pol, y) == pytest.approx(prob(ref, y) * math.exp(reward[y] / beta) / Z, rel=1e-10)

def test_potential_based_shaping_preserves_advantages():              # Proposition 10
    m, pi = random_mdp(5, 3, 0.9, seed=0), rand_pi(5, 3, 3)
    phi = torch.randn(5, **f64)
    m2 = dataclasses.replace(m, r=m.r + m.gamma * m.P @ phi - phi[:, None])
    V, V2 = policy_evaluation(m, pi), policy_evaluation(m2, pi)
    torch.testing.assert_close(V2, V - phi)
    torch.testing.assert_close(q_values(m2, V2) - V2[:, None], q_values(m, V) - V[:, None])

def test_importance_weight_second_moment():                           # Proposition 9
    pi, b = torch.tensor([0.7, 0.2, 0.1], **f64), torch.full((3,), 1 / 3, **f64)
    chi2 = ((pi - b) ** 2 / b).sum().item()
    for H in (1, 5, 20):
        assert is_second_moment(pi, b, H) == pytest.approx((1 + chi2) ** H)
