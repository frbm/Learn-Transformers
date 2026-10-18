"""R1 §1-4. Exact tabular quantities in float64. Policies are (S, A) row-stochastic tensors.

Every function also accepts a leading batch axis of MDPs (stack_mdps). Batched solves remove the Python loop."""
from dataclasses import dataclass

import torch
from torch import Tensor

f64 = dict(dtype=torch.float64)


@dataclass
class MDP:
    P: Tensor        # (S, A, S) transition kernel
    r: Tensor        # (S, A) rewards in [0, 1]
    gamma: float
    rho: Tensor      # (S,) initial distribution, full support

    @property
    def S(self) -> int:
        return self.r.shape[-2]

    @property
    def A(self) -> int:
        return self.r.shape[-1]


def stack_mdps(ms: list[MDP]) -> MDP:
    assert len({m.gamma for m in ms}) == 1
    return MDP(*(torch.stack([getattr(m, k) for m in ms]) for k in ("P", "r")), ms[0].gamma,
               torch.stack([m.rho for m in ms]))


def random_mdp(S: int, A: int, gamma: float, seed: int = 0) -> MDP:
    g = torch.Generator().manual_seed(seed)
    # Scale the logits by 2. Then the rows are far from uniform, and the actions really differ.
    P = torch.softmax(2 * torch.randn(S, A, S, generator=g, **f64), -1)
    r = torch.rand(S, A, generator=g, **f64)
    rho = torch.softmax(torch.randn(S, generator=g, **f64), 0)
    return MDP(P, r, gamma, rho)


def _p_pi(m: MDP, pi: Tensor) -> tuple[Tensor, Tensor]:
    return torch.einsum("...sa,...sat->...st", pi, m.P), (pi * m.r).sum(-1)


def policy_evaluation(m: MDP, pi: Tensor) -> Tensor:
    P_pi, r_pi = _p_pi(m, pi)
    return torch.linalg.solve(torch.eye(m.S, **f64) - m.gamma * P_pi, r_pi)


def q_values(m: MDP, V: Tensor) -> Tensor:
    return m.r + m.gamma * torch.einsum("...sat,...t->...sa", m.P, V)


def occupancy(m: MDP, pi: Tensor) -> Tensor:
    P_pi, _ = _p_pi(m, pi)
    # d^T = (1 - gamma) rho^T (I - gamma P_pi)^-1. Solve with the transpose. Do not invert.
    return (1 - m.gamma) * torch.linalg.solve((torch.eye(m.S, **f64) - m.gamma * P_pi).mT, m.rho)


def objective(m: MDP, pi: Tensor) -> Tensor:
    return (m.rho * policy_evaluation(m, pi)).sum(-1)


def greedy(Q: Tensor) -> Tensor:
    return torch.nn.functional.one_hot(Q.argmax(-1), Q.shape[-1]).to(Q.dtype)


def value_iteration(m: MDP, tol: float = 1e-12) -> tuple[Tensor, Tensor]:
    V = torch.zeros(m.S, **f64)
    while True:
        V_new = q_values(m, V).max(-1).values
        done = (V_new - V).abs().max() < tol
        V = V_new
        if done:
            return V, greedy(q_values(m, V))


def policy_iteration(m: MDP, trace: list | None = None) -> tuple[Tensor, int]:
    """Exact evaluation, then greedy improvement. `trace` receives J after each evaluation."""
    act = m.r.argmax(-1)
    for it in range(1, 10_000):
        pi = torch.nn.functional.one_hot(act, m.A).to(m.r.dtype)
        V = policy_evaluation(m, pi)
        if trace is not None:
            trace.append((m.rho * V).sum(-1))
        Q = q_values(m, V)
        # Keep the current action on ties. Otherwise two equal actions can alternate forever.
        keep = Q.gather(-1, act[..., None])[..., 0] >= Q.max(-1).values - 1e-12
        new = torch.where(keep, act, Q.argmax(-1))
        if torch.equal(new, act):
            return pi, it
        act = new
    raise RuntimeError("policy iteration did not terminate")


def softmax_policy(theta: Tensor) -> Tensor:
    return torch.softmax(theta, -1)


def _pi_A_d(m: MDP, theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    pi = softmax_policy(theta)
    V = policy_evaluation(m, pi)
    return pi, q_values(m, V) - V[..., None], occupancy(m, pi)


def pg_exact(m: MDP, theta: Tensor) -> Tensor:
    pi, A, d = _pi_A_d(m, theta)
    return d[..., None] * pi * A / (1 - m.gamma)


def fisher(m: MDP, theta: Tensor) -> Tensor:
    """F = sum_s d(s) sum_a pi psi psi^T. Block diagonal: d(s) (diag(pi_s) - pi_s pi_s^T) per state."""
    pi, d = softmax_policy(theta), occupancy(m, softmax_policy(theta))
    blocks = d[:, None, None] * (torch.diag_embed(pi) - pi[:, :, None] * pi[:, None, :])
    return torch.block_diag(*blocks)


def npg_direction(m: MDP, theta: Tensor) -> Tensor:
    # F is singular: each state block kills the constant vector. The default pinv cut-off
    # (S*A*eps relative) drops these exact zeros, but keeps the small true eigenvalues
    # d(s) pi (1 - pi) of near-deterministic policies. A larger rtol cuts them, so NPG stalls too.
    F = fisher(m, theta)
    return (torch.linalg.pinv(F, hermitian=True) @ pg_exact(m, theta).flatten()).view_as(theta)
