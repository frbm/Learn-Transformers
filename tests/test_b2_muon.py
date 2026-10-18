# tests/test_b2_muon.py
import pytest, torch
from lm.train.optim import newton_schulz, Muon, spectral_norm_estimate

DEV = "cuda" if torch.cuda.is_available() else "cpu"

def svd_parts(G):
    U, S, Vh = torch.linalg.svd(G.double(), full_matrices=False)
    return U, S, Vh

# Change from the page: the square case is a strict xfail. A square Gaussian matrix has
# sigma_min / ||G||_F ~ 1e-5. Five quintic steps multiply a small singular value by at most
# 3.4445^5 ~ 490, so diag(C).min() ~ 0.01-0.1 for every correct implementation.
# test_quintic_matches_scalar_iteration below checks the square case against the exact prediction.
SQUARE = pytest.param((256, 256), marks=pytest.mark.xfail(strict=True, reason="sigma_min of a square Gaussian is ~0"))

@pytest.mark.parametrize("shape", [SQUARE, (256, 512), (512, 256)])
def test_quintic_keeps_singular_vectors_and_flattens(shape):     # Proposition 3, default coefficients
    G = torch.randn(*shape, device=DEV)
    O = newton_schulz(G).double()
    U, _, Vh = svd_parts(G)
    C = U.mT @ O @ Vh.mT                                           # O expressed in G's singular bases
    s = torch.diagonal(C)
    assert s.min() > 0.5 and s.max() < 1.5
    assert (C - torch.diag(s)).norm() / s.norm() < 0.1             # off-diagonal = rotated singular vectors

def test_cubic_converges_to_polar_factor():                         # §4, exact in float64
    G = torch.randn(128, 256, device=DEV, dtype=torch.float64)
    U, _, Vh = svd_parts(G)
    O = newton_schulz(G, steps=40, coeffs=(1.5, -0.5, 0.0), dtype=torch.float64)
    torch.testing.assert_close(O, U @ Vh, atol=1e-6, rtol=0)

def test_scale_invariance():                                        # Exercise 1
    G = torch.randn(256, 384, device=DEV)
    torch.testing.assert_close(newton_schulz(G).float(), newton_schulz(1e-3 * G).float(), atol=2e-2, rtol=0)

def test_power_iteration_matches_svd():
    W = torch.randn(300, 200, device=DEV)
    assert abs(spectral_norm_estimate(W) / torch.linalg.matrix_norm(W, ord=2).item() - 1) < 1e-2

def test_muon_step_size():                                          # one step from zero: ||dW||_2 = lr * scale * ~1
    W = torch.nn.Parameter(torch.zeros(256, 1024, device=DEV))
    opt = Muon([W], lr=0.1, momentum=0.0, nesterov=False)
    W.grad = torch.randn_like(W)
    opt.step()
    assert 0.05 < torch.linalg.matrix_norm(W.detach(), ord=2).item() < 0.15


# Added tests.

def phi_iter(s, steps=5, coeffs=(3.4445, -4.7750, 2.0315)):
    a, b, c = coeffs
    for _ in range(steps):
        s = a * s + b * s**3 + c * s**5
    return s

@pytest.mark.parametrize("shape", [(256, 256), (128, 384)])
def test_quintic_matches_scalar_iteration(shape):                  # Proposition 3 as an oracle, square case included
    G = torch.randn(*shape, device=DEV, dtype=torch.float64)
    U, S, Vh = svd_parts(G)
    O = newton_schulz(G, dtype=torch.float64)
    torch.testing.assert_close(O, U @ torch.diag(phi_iter(S / S.norm())) @ Vh, atol=1e-9, rtol=0)

def test_tall_matrix_scale():                                       # ||dW||_2 = lr * sqrt(d_out / d_in) * phi^5(s)
    W = torch.nn.Parameter(torch.zeros(1024, 256, device=DEV))
    U, _, Vh = svd_parts(torch.randn(1024, 256, device=DEV))
    W.grad = (U @ Vh).float()                                       # all singular values 1, so 1/16 after normalising
    Muon([W], lr=0.1, momentum=0.0, nesterov=False).step()
    expected = 0.1 * 2 * phi_iter(torch.tensor(1 / 16, dtype=torch.float64)).item()
    assert abs(torch.linalg.matrix_norm(W.detach(), ord=2).item() / expected - 1) < 0.05            # bf16 iteration

def test_nesterov_momentum_matches_manual():                        # two code paths: optimiser vs hand-written recursion
    torch.manual_seed(0)
    W = torch.nn.Parameter(torch.randn(64, 96, device=DEV))
    W_ref, M = W.detach().clone(), torch.zeros_like(W)
    opt = Muon([W], lr=0.05, momentum=0.9, weight_decay=0.1)
    for _ in range(3):
        G = torch.randn_like(W)
        W.grad = G
        opt.step()
        M = 0.9 * M + G
        W_ref = W_ref * (1 - 0.05 * 0.1) - 0.05 * newton_schulz(G + 0.9 * M).float()
    torch.testing.assert_close(W.detach(), W_ref, atol=1e-5, rtol=0)

def test_build_optimizers_partition():
    from lm.model import Config, GPT
    from lm.model.mup import init_weights
    from lm.train.optim import build_optimizers
    cfg = Config(vocab_size=256, d_model=128, n_layers=2, n_heads=2, n_kv_heads=1, max_seq_len=32,
                 parametrization="mup", base_width=128)
    m = GPT(cfg).to(DEV)
    init_weights(m)
    muon, adam = build_optimizers(m, 0.02, 3e-3, 0.0)
    ids = [id(p) for o in (muon, adam) for g in o.param_groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(m.parameters()))
    assert all(p.ndim == 2 for g in muon.param_groups for p in g["params"])
    assert {g["kind"] for g in adam.param_groups} == {"embedding", "readout", "vector"}
