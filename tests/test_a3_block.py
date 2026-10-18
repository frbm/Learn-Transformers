import pytest
import torch
import torch.nn.functional as F

from lm.model import GPT, Config, expected_params

from lm.model.attention import attention_core, attention_sdpa

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CFG = Config(vocab_size=1024, d_model=256, n_layers=4, n_heads=4, n_kv_heads=2, max_seq_len=128)


def oracle(q, k, v):
    g = q.shape[1] // k.shape[1]
    return F.scaled_dot_product_attention(q, k.repeat_interleave(g, 1), v.repeat_interleave(g, 1), is_causal=True)


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-5), (torch.bfloat16, 2e-2)])
def test_attention_matches_sdpa(dtype, tol):
    q = torch.randn(2, 8, 64, 32, device=DEV, dtype=dtype)
    k = torch.randn(2, 2, 64, 32, device=DEV, dtype=dtype)
    v = torch.randn(2, 2, 64, 32, device=DEV, dtype=dtype)
    torch.testing.assert_close(attention_core(q, k, v), oracle(q, k, v), atol=tol, rtol=tol)


def test_offset_mask_naive_vs_fast():
    q = torch.randn(2, 8, 8, 32, device=DEV)
    k = torch.randn(2, 2, 24, 32, device=DEV)
    v = torch.randn(2, 2, 24, 32, device=DEV)
    torch.testing.assert_close(attention_core(q, k, v, q_offset=16), attention_sdpa(q, k, v, q_offset=16), atol=1e-5, rtol=1e-5)


def test_softmax_jacobian_is_covariance():
    z = torch.randn(7, dtype=torch.float64)
    w = torch.softmax(z, 0)
    J = torch.autograd.functional.jacobian(lambda u: torch.softmax(u, 0), z)
    torch.testing.assert_close(J, torch.diag(w) - torch.outer(w, w))


def test_param_count():
    assert sum(p.numel() for p in GPT(CFG).parameters()) == expected_params(CFG)


def test_param_count_tier0():
    cfg = Config(vocab_size=16448, d_model=256, n_layers=4, n_heads=4, n_kv_heads=2, max_seq_len=512)
    assert abs(expected_params(cfg) - 7.16e6) < 0.01e6


@pytest.mark.parametrize("impl", ["naive", "sdpa"])
def test_causality(impl):
    m = GPT(Config(**{**CFG.__dict__, "attn_impl": impl})).to(DEV).eval()
    x = torch.randint(0, CFG.vocab_size, (1, 32), device=DEV)
    y = x.clone(); y[0, 20:] = torch.randint(0, CFG.vocab_size, (12,), device=DEV)
    with torch.no_grad():
        torch.testing.assert_close(m(x)[:, :20], m(y)[:, :20])


@pytest.mark.parametrize("impl", ["naive", "sdpa"])
def test_kv_cache_matches_full_forward(impl):
    m = GPT(Config(**{**CFG.__dict__, "attn_impl": impl})).to(DEV).eval()
    x = torch.randint(0, CFG.vocab_size, (2, 40), device=DEV)
    with torch.no_grad():
        full = m(x)
        cache = m.init_cache(batch_size=2)
        parts = [m(x[:, :16], cache=cache), m(x[:, 16:24], cache=cache)]
        parts += [m(x[:, t:t + 1], cache=cache) for t in range(24, 40)]
    torch.testing.assert_close(torch.cat(parts, 1), full, atol=1e-4, rtol=1e-4)


def test_greedy_generation_identical():
    m = GPT(CFG).to(DEV).eval()
    p = torch.randint(0, CFG.vocab_size, (1, 8), device=DEV)
    with torch.no_grad():
        assert torch.equal(m.generate(p, 32, use_cache=True), m.generate(p, 32, use_cache=False))
