import torch
import torch.nn.functional as F
from torch.func import functional_call
from lm.analysis.curvature import hvp, ggn_vp, lanczos, hutchinson_trace

torch.manual_seed(0)
NET = torch.nn.Sequential(torch.nn.Linear(8, 12), torch.nn.Tanh(), torch.nn.Linear(12, 5)).double()   # 173 parameters
PARAMS = {k: v.detach() for k, v in NET.named_parameters()}
X, Y = torch.randn(32, 8, dtype=torch.float64), torch.randint(0, 5, (32,))
THETA = torch.cat([v.flatten() for v in PARAMS.values()])

def unflat(t):
    out, i = {}, 0
    for k, v in PARAMS.items():
        out[k] = t[i:i + v.numel()].view_as(v); i += v.numel()
    return out

def logits_fn(t): return functional_call(NET, unflat(t), (X,))
def loss_fn(t): return F.cross_entropy(logits_fn(t), Y)

def dense_ggn():
    J = torch.autograd.functional.jacobian(logits_fn, THETA)            # (N, K, P)
    p = torch.softmax(logits_fn(THETA), -1)
    Lam = torch.diag_embed(p) - p.unsqueeze(-1) * p.unsqueeze(-2)       # (N, K, K)
    return torch.einsum("nkp,nkl,nlq->pq", J, Lam, J) / X.shape[0]

def test_hvp_matches_dense_hessian():
    v = torch.randn_like(THETA)
    torch.testing.assert_close(hvp(loss_fn, THETA, v), torch.autograd.functional.hessian(loss_fn, THETA) @ v)

def test_ggn_vp_matches_dense():
    v = torch.randn_like(THETA)
    torch.testing.assert_close(ggn_vp(logits_fn, THETA, v), dense_ggn() @ v)

def test_ggn_is_psd_and_low_rank():                                     # Proposition 1
    ev = torch.linalg.eigvalsh(dense_ggn())
    assert ev.min() > -1e-10 and (ev > 1e-10).sum() <= X.shape[0] * (5 - 1)

def test_residual_term_is_indefinite():                                 # Proposition 2
    ev = torch.linalg.eigvalsh(torch.autograd.functional.hessian(loss_fn, THETA) - dense_ggn())
    assert ev.min() < 0 < ev.max()

def test_lanczos_top_eigenvalues():
    Q, _ = torch.linalg.qr(torch.randn(300, 300, dtype=torch.float64))
    spec = torch.cat([torch.tensor([50., 40., 30.], dtype=torch.float64), torch.rand(297, dtype=torch.float64)])
    A = Q @ torch.diag(spec) @ Q.T
    torch.testing.assert_close(lanczos(lambda v: A @ v, dim=300, k=40)[:3], spec[:3], atol=1e-6, rtol=1e-6)

def test_hutchinson_within_five_sigma():                                # Proposition 5
    A = torch.randn(100, 100, dtype=torch.float64); A = A + A.T
    n = 20_000
    sd = (2 * (A.pow(2).sum() - A.diag().pow(2).sum())).sqrt().item() / n ** 0.5
    assert abs(hutchinson_trace(lambda v: A @ v, dim=100, n_samples=n) - A.trace().item()) < 5 * sd
