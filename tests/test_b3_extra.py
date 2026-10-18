"""B3 extra tests: the model wrappers, chunking, and the failure mode of Lanczos without re-orthogonalisation."""
import functools

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from lm.analysis.curvature import Curvature, FlatModel, ggn_vp, hvp, lanczos, preconditioned
from lm.model import GPT, Config
close = functools.partial(torch.testing.assert_close, rtol=1e-4, atol=1e-6)


def tiny(attn_impl="naive"):
    torch.manual_seed(0)
    cfg = Config(vocab_size=64, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1, max_seq_len=16, attn_impl=attn_impl)
    m = GPT(cfg).double().eval()
    x = torch.randint(0, 64, (6, 16), generator=torch.Generator().manual_seed(1))
    return m, x[:, :-1], x[:, 1:]


def test_chunked_products_equal_full_batch():                 # ratio of sums over unequal chunks
    m, x, y = tiny()
    fm = FlatModel(m)
    theta, v = fm.flat().double(), torch.randn(fm.dim, dtype=torch.float64)
    full, chunked = Curvature(fm, x, y, chunk=len(x)), Curvature(fm, x, y, chunk=4)
    close(chunked.hvp(theta, v), full.hvp(theta, v))
    close(chunked.ggn_vp(theta, v), full.ggn_vp(theta, v))
    (l1, g1), (l2, g2) = chunked.value_and_grad(theta), full.value_and_grad(theta)
    close(g1, g2)
    assert abs(l1 - l2) < 1e-6


def test_sdpa_math_path_matches_naive():                      # two code paths, one number
    (m1, x, y), (m2, _, _) = tiny("naive"), tiny("sdpa")
    f1, f2 = FlatModel(m1), FlatModel(m2)
    theta, v = f1.flat().double(), torch.randn(f1.dim, dtype=torch.float64)
    with sdpa_kernel(SDPBackend.MATH):
        close(hvp(f2.loss_fn(x, y), theta, v), hvp(f1.loss_fn(x, y), theta, v))
        close(ggn_vp(f2.logits_fn(x), theta, v), ggn_vp(f1.logits_fn(x), theta, v))


def test_measurement_leaves_grad_buffers_alone():
    m, x, y = tiny()
    fm = FlatModel(m)
    Curvature(fm, x, y, chunk=3).hvp(fm.flat().double(), torch.randn(fm.dim, dtype=torch.float64))
    assert all(p.grad is None for p in m.parameters())


def test_lanczos_without_reorth_makes_ghosts():               # the check must be able to fail
    torch.manual_seed(0)
    Q, _ = torch.linalg.qr(torch.randn(300, 300, dtype=torch.float64))
    spec = torch.cat([torch.tensor([50.0]), torch.rand(299, dtype=torch.float64)])
    A = Q @ torch.diag(spec) @ Q.T
    near = lambda r: (lanczos(lambda v: A @ v, 300, 120, reorth=r) - 50).abs().lt(1e-3).sum().item()
    assert near(True) == 1 and near(False) > 1


def test_preconditioned_spectrum_equals_dense():
    torch.manual_seed(0)
    B = torch.randn(40, 40, dtype=torch.float64); A = B @ B.T
    p = torch.rand(40, dtype=torch.float64) + 0.1
    dense = torch.linalg.eigvals(torch.diag(1 / p) @ A).real.sort(descending=True).values
    ritz = lanczos(preconditioned(lambda v: A @ v, p.rsqrt()), 40, 40)
    torch.testing.assert_close(ritz[:3], dense[:3])



def test_cached_curvature_matches_uncached():                 # two code paths: per-chunk caches vs recomputation
    m, x, y = tiny()
    fm = FlatModel(m)
    theta = fm.flat().double()
    cached, plain = Curvature(fm, x, y, chunk=4, cache=True), Curvature(fm, x, y, chunk=4, cache=False)
    for seed in range(2):                                      # the second round reuses the caches
        v = torch.randn(fm.dim, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))
        close(cached.hvp(theta, v), plain.hvp(theta, v))
        close(cached.ggn_vp(theta, v), plain.ggn_vp(theta, v))


def test_cache_sees_in_place_updates():                       # the cache keys on the version counter
    m, x, y = tiny()
    fm = FlatModel(m)
    theta = fm.flat().double()
    cached, plain = Curvature(fm, x, y, chunk=4, cache=True), Curvature(fm, x, y, chunk=4, cache=False)
    v = torch.randn(fm.dim, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    cached.hvp(theta, v)
    theta.sub_(0.1 * v)                                        # same tensor object, new values
    close(cached.hvp(theta, v), plain.hvp(theta, v))
    close(cached.ggn_vp(theta, v), plain.ggn_vp(theta, v))
