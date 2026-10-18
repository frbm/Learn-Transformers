"""R1 §3 and §6. Monte Carlo estimators, vectorised over trajectories."""
import torch
from torch import Tensor

from lm.rl.tabular import MDP, softmax_policy


def sample_trajectories(m: MDP, pi: Tensor, n: int, H: int, seed: int = 0) -> dict[str, Tensor]:
    """Vectorise over n trajectories. Loop over time only."""
    g = torch.Generator().manual_seed(seed)
    states = torch.empty(n, H + 1, dtype=torch.long)
    actions = torch.empty(n, H, dtype=torch.long)
    states[:, 0] = torch.multinomial(m.rho.expand(n, -1), 1, generator=g)[:, 0]
    for t in range(H):
        s = states[:, t]
        a = torch.multinomial(pi[s], 1, generator=g)[:, 0]
        actions[:, t] = a
        states[:, t + 1] = torch.multinomial(m.P[s, a], 1, generator=g)[:, 0]
    return {"states": states, "actions": actions, "rewards": m.r[states[:, :-1], actions]}


def discounts(gamma: float, H: int) -> Tensor:
    return gamma ** torch.arange(H, dtype=torch.float64)


def weighted_score(theta: Tensor, traj: dict[str, Tensor], coef: Tensor) -> Tensor:
    """sum_t coef[:, t] * grad log pi(a_t | s_t) for the tabular softmax -> (n, S, A)."""
    S, A = theta.shape
    s, a = traj["states"][:, :-1], traj["actions"]
    pi = softmax_policy(theta)
    # Softmax score in row s: onehot(a) - pi(.|s). Scatter-add it into row s_t of each estimate.
    psi = torch.nn.functional.one_hot(a, A).to(theta.dtype) - pi[s]
    out = torch.zeros(s.shape[0], S, A, dtype=theta.dtype)
    return out.scatter_add_(1, s[..., None].expand(-1, -1, A), coef[..., None] * psi)


def returns_to_go(rewards: Tensor, gamma: float) -> Tensor:
    H = rewards.shape[1]
    k = torch.arange(H)
    # M[k, t] = gamma^(k - t) for k >= t. One matmul replaces the reverse loop.
    M = torch.where(k[:, None] >= k[None, :], gamma ** (k[:, None] - k[None, :]).double(), 0.0)
    return rewards @ M


def reinforce_grad(theta, traj, gamma: float, reward_to_go: bool = True, baseline: Tensor | None = None) -> Tensor:
    r = traj["rewards"]
    H = r.shape[1]
    disc = discounts(gamma, H)
    b = baseline[traj["states"][:, :-1]] if baseline is not None else torch.zeros_like(r)
    if reward_to_go:
        coef = disc * (returns_to_go(r, gamma) - b)
    else:
        # The whole return G_0 already carries gamma^t. Discount only the baseline.
        coef = (r * disc).sum(1, keepdim=True) - disc * b
    return weighted_score(theta, traj, coef)


def gae(rewards: Tensor, values: Tensor, gamma: float, lam: float) -> Tensor:
    delta = rewards + gamma * values[:, 1:] - values[:, :-1]
    return returns_to_go(delta, gamma * lam)


def is_second_moment(pi: Tensor, b: Tensor, H: int) -> float:
    """E_b[W^2] = (sum_a pi^2 / b)^H = (1 + chi2(pi || b))^H in a single state."""
    return ((pi ** 2 / b).sum() ** H).item()
