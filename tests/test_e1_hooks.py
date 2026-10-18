import pytest, torch
from lm.model import Config, GPT
from lm.interp.hooks import run_with_cache
from lm.interp.circuits import (head_outputs, residual_components, dla, qk_pattern, copying_score,
                             prefix_matching_score, patch, linear_patch)

f64 = dict(dtype=torch.float64)

@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    cfg = Config(vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=16)
    model = GPT(cfg).double().eval()
    idx = torch.randint(0, 64, (3, 16))
    logits, cache = run_with_cache(model, idx)
    return cfg, model, idx, logits, cache

def test_cache_and_cleanup(setup):
    cfg, model, idx, logits, cache = setup
    pat = cache["blocks.0.attn.hook_pattern"]
    torch.testing.assert_close(pat.sum(-1), torch.ones_like(pat.sum(-1)))
    assert torch.all(pat.triu(1) == 0)
    assert cache["blocks.1.attn.hook_z"].shape == (3, 16, 4, 8)
    assert all(len(m._forward_hooks) == 0 for m in model.modules())          # no hook left behind

def test_residual_is_a_sum(setup):                                            # Proposition 1
    cfg, model, idx, logits, cache = setup
    torch.testing.assert_close(sum(residual_components(model, cache).values()), cache["blocks.1.hook_resid_post"])

def test_heads_sum_to_attention_output(setup):
    cfg, model, idx, logits, cache = setup
    torch.testing.assert_close(head_outputs(model, cache, 1).sum(2), cache["blocks.1.hook_attn_out"])

def test_direct_logit_attribution_is_exact(setup):                            # Proposition 1
    cfg, model, idx, logits, cache = setup
    direction = torch.zeros(cfg.vocab_size, **f64); direction[7] = 1.0
    contrib = dla(model, cache, residual_components(model, cache), direction)
    torch.testing.assert_close(sum(contrib.values()), logits[..., 7])

def test_qk_circuit_reproduces_the_pattern(setup):                            # Proposition 2, with RoPE
    cfg, model, idx, logits, cache = setup
    for layer, head in [(0, 0), (1, 3)]:
        torch.testing.assert_close(qk_pattern(model, cache, layer, head), cache[f"blocks.{layer}.attn.hook_pattern"][:, head])

def test_copying_score_small_matrix_trick():                                  # Proposition 3
    torch.manual_seed(1)
    W_E, W_OV, W_U = torch.randn(50, 8, **f64), torch.randn(8, 8, **f64), torch.randn(8, 50, **f64)
    lam = torch.linalg.eigvals(W_E @ W_OV @ W_U)
    assert copying_score(W_E, W_OV, W_U) == pytest.approx((lam.real.sum() / lam.abs().sum()).item(), rel=1e-6)
    assert copying_score(W_E, torch.eye(8, **f64), W_E.T) == pytest.approx(1.0)

def test_prefix_matching_score():
    n, T = 6, 12
    perfect = torch.zeros(T, T); perfect[:n, 0] = 1
    for i in range(n, T): perfect[i, i - n + 1] = 1                          # attend to the token after the first occurrence
    assert prefix_matching_score(perfect, n) == pytest.approx(1.0)
    uniform = torch.tril(torch.ones(T, T)); uniform /= uniform.sum(-1, keepdim=True)
    assert prefix_matching_score(uniform, n) < 0.2

def test_patching_identities(setup):
    cfg, model, idx, logits, cache = setup
    corrupt = torch.randint(0, 64, idx.shape, generator=torch.Generator().manual_seed(5))
    metric = lambda lg: lg[:, -1, 7].sum()
    _, ccache = run_with_cache(model, corrupt)
    with torch.no_grad():
        base = metric(model(corrupt)).item()
    assert patch(model, corrupt, "blocks.1.hook_attn_out", ccache["blocks.1.hook_attn_out"], metric) == pytest.approx(base)
    assert patch(model, corrupt, "blocks.1.hook_resid_pre", cache["blocks.1.hook_resid_pre"], metric) == pytest.approx(metric(logits).item())

def test_attribution_patching_is_first_order(setup):                          # Proposition 6
    cfg, model, idx, logits, cache = setup
    metric = lambda lg: lg[:, -1, 7].sum()
    a = cache["blocks.1.hook_resid_mid"]
    delta = torch.randn(a.shape, generator=torch.Generator().manual_seed(3), **f64)
    errs = [abs(patch(model, idx, "blocks.1.hook_resid_mid", a + e * delta, metric)
                - linear_patch(model, idx, "blocks.1.hook_resid_mid", a + e * delta, metric)) for e in (1e-2, 5e-3)]
    assert 3.0 < errs[0] / errs[1] < 5.0                                      # error is O(eps^2)


# Added tests.

def test_hooked_forward_matches_the_original_model():
    torch.manual_seed(0)
    cfg = Config(vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=16)
    ref, hooked = GPT(cfg).eval(), GPT(cfg).eval()
    hooked.load_state_dict(ref.state_dict())
    idx = torch.randint(0, 64, (2, 16))
    logits, _ = run_with_cache(hooked, idx)                                   # explicit pattern path
    with torch.no_grad():
        torch.testing.assert_close(logits, ref(idx), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(hooked(idx), ref(idx), atol=1e-5, rtol=1e-5)   # fused path, no hooks
        torch.testing.assert_close(hooked.generate(idx[:, :4], 6), ref.generate(idx[:, :4], 6))   # KV cache path

def test_hooks_are_removed_after_an_exception(setup):
    from lm.interp.hooks import hooks
    cfg, model, idx, logits, cache = setup
    with pytest.raises(RuntimeError):
        with hooks(model, {"blocks.0.hook_resid_pre": lambda o: None}):
            raise RuntimeError
    assert all(len(m._forward_hooks) == 0 for m in model.modules())

def test_attribution_grid_matches_exact_patching_for_small_differences(setup):
    from lm.interp.circuits import activation_patch, attribution_patch
    cfg, model, idx, logits, cache = setup
    metric = lambda lg: lg[:, -1, 7].sum()
    corrupt = idx.clone(); corrupt[:, 3] = (corrupt[:, 3] + 1) % 64
    # The stream right after the swapped token is far from linear (Proposition 6). Test later nodes.
    for name in ["blocks.0.hook_mlp_out", "blocks.0.attn.hook_z", "blocks.1.hook_resid_mid"]:
        exact = activation_patch(model, idx, corrupt, name, metric)
        approx = attribution_patch(model, idx, corrupt, [name], metric)[name]
        assert exact.shape == approx.shape
        assert torch.corrcoef(torch.stack([exact.flatten(), approx.flatten()]))[0, 1] > 0.9

def test_attribution_error_is_second_order_at_the_scale_of_the_activation(setup):
    # The given test perturbs by e * N(0, I). The stream of model.GPT at init has norm ~0.1, so e = 1e-2 is
    # not small and third-order terms show. Scale the perturbation to the activation norm instead.
    cfg, model, idx, logits, cache = setup
    metric = lambda lg: lg[:, -1, 7].sum()
    a = cache["blocks.1.hook_resid_mid"]
    delta = torch.randn(a.shape, generator=torch.Generator().manual_seed(3), **f64)
    delta *= a.norm(dim=-1, keepdim=True) / delta.norm(dim=-1, keepdim=True)
    errs = [abs(patch(model, idx, "blocks.1.hook_resid_mid", a + e * delta, metric)
                - linear_patch(model, idx, "blocks.1.hook_resid_mid", a + e * delta, metric)) for e in (1e-2, 5e-3)]
    assert 3.0 < errs[0] / errs[1] < 5.0
