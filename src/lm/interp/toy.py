"""E2 toy models of superposition (Elhage et al. 2022): x_hat = ReLU(W^T W x + b)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ToyModel(nn.Module):
    def __init__(self, n: int, m: int):
        super().__init__()
        self.W = nn.Parameter(nn.init.xavier_normal_(torch.empty(m, n)))
        self.b = nn.Parameter(torch.zeros(n))

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x @ self.W.T @ self.W + self.b)


def sample_features(batch: int, n: int, p: float | Tensor, generator=None, dtype=torch.float32) -> Tensor:
    """Each feature is active with probability p, then uniform on [0, 1]."""
    u = torch.rand(2, batch, n, generator=generator, dtype=dtype)
    return u[1] * (u[0] < p)


def toy_expected_loss(W: Tensor, b: Tensor, p: float, n_samples: int = 200_000, seed: int = 0) -> float:
    """Monte Carlo E sum_i (x_i - x_hat_i)^2 with unit importances."""
    x = sample_features(n_samples, W.shape[1], p, torch.Generator().manual_seed(seed), W.dtype)
    return ((x - F.relu(x @ W.T @ W + b)) ** 2).sum(1).mean().item()


def feature_dimensionality(W: Tensor) -> Tensor:
    """D_i = ||W_i||^2 / sum_j (W_i_hat . W_j)^2 for the columns W_i of W (m, n). 0 for a dropped feature."""
    norm = W.norm(dim=0)
    G = (W / norm.clamp_min(1e-12)).T @ W
    return torch.where(norm > 1e-6, norm**2 / (G**2).sum(1).clamp_min(1e-24), torch.zeros_like(norm))


def train_toy(n: int, m: int, p: float, importance: Tensor | None = None, steps: int = 5000, batch: int = 1024,
              lr: float = 1e-3, seed: int = 0) -> ToyModel:
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = ToyModel(n, m)
    I = torch.ones(n) if importance is None else importance
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    for _ in range(steps):
        x = sample_features(batch, n, p, g)
        loss = (I * (x - model(x)) ** 2).sum(1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model
