import itertools, pytest, torch
from lm.interp.sae import SAE, topk_activation, soft_threshold, hard_threshold, normalize_decoder_, remove_parallel_grad_, fvu, loss_recovered
from lm.interp.toy import toy_expected_loss, feature_dimensionality

f64 = dict(dtype=torch.float64)

def test_topk_is_the_projection_onto_k_sparse_nonnegative_vectors():
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        u, k = torch.randn(8, generator=g, **f64), 3
        best = min((torch.zeros(8, **f64).index_put((torch.tensor(S),), torch.relu(u[list(S)]))
                    for S in itertools.combinations(range(8), k)), key=lambda z: ((z - u) ** 2).sum().item())
        torch.testing.assert_close(topk_activation(u, k), best)

def test_activations_are_proximal_operators():                                # Proposition 5
    s, lam = torch.linspace(-2, 2, 81, **f64), 0.7
    grid = torch.cat([torch.linspace(-3, 3, 60001, **f64), torch.zeros(1, **f64)])
    soft = grid.clamp(min=0)[((s[:, None] - grid.clamp(min=0)) ** 2 + lam * grid.clamp(min=0)).argmin(1)]
    hard = grid[((s[:, None] - grid) ** 2 + lam * (grid != 0)).argmin(1)]
    torch.testing.assert_close(soft_threshold(s, lam), soft, atol=2e-4, rtol=0)
    torch.testing.assert_close(hard_threshold(s, lam), hard, atol=2e-4, rtol=0)

def test_decoder_norm_and_tangent_gradient():
    sae = SAE(16, 64, k=4, dtype=torch.float64)
    with torch.no_grad(): sae.W_dec.mul_(3.0)
    normalize_decoder_(sae)
    torch.testing.assert_close(sae.W_dec.norm(dim=0), torch.ones(64, **f64))
    x = torch.randn(32, 16, **f64)
    ((sae(x)[0] - x) ** 2).sum().backward()
    remove_parallel_grad_(sae)
    torch.testing.assert_close((sae.W_dec.grad * sae.W_dec).sum(0), torch.zeros(64, **f64), atol=1e-10, rtol=0)

def test_metrics():
    x = torch.randn(100, 8, **f64)
    assert fvu(x, x).item() == 0.0 and fvu(x, x.mean(0).expand_as(x)).item() == pytest.approx(1.0)
    assert loss_recovered(ce_clean=3.0, ce_sae=3.5, ce_zero=8.0) == pytest.approx(0.9)

def test_superposition_beats_a_dedicated_dimension_when_features_are_sparse():   # Proposition 4
    anti = torch.tensor([[1.0, -1.0]], **f64)
    for p in (0.2, 0.9):
        assert toy_expected_loss(anti, torch.zeros(2, **f64), p) == pytest.approx(p * p / 3, rel=0.03)
        dedicated = toy_expected_loss(torch.tensor([[1.0, 0.0]], **f64), torch.tensor([0.0, p / 2], **f64), p)
        assert dedicated == pytest.approx(p / 3 - p * p / 4, rel=0.03)
        assert (p * p / 3 < dedicated) == (p < 4 / 7)

def test_feature_dimensionality():
    assert torch.allclose(feature_dimensionality(torch.eye(4, **f64)), torch.ones(4, **f64))
    assert torch.allclose(feature_dimensionality(torch.tensor([[1.0, -1.0]], **f64)), torch.full((2,), 0.5, **f64))

def test_topk_sae_recovers_a_planted_dictionary():                           # identifiability, in practice
    g = torch.Generator().manual_seed(0)
    d, m_true, k = 32, 64, 3
    D = torch.randn(d, m_true, generator=g); D /= D.norm(dim=0, keepdim=True)
    def batch(n):
        idx = torch.rand(n, m_true, generator=g).argsort(1)[:, :k]
        a = torch.zeros(n, m_true).scatter(1, idx, torch.rand(n, k, generator=g) + 0.5)
        return a @ D.t()
    torch.manual_seed(0)
    sae = SAE(d, 2 * m_true, k=k)
    opt = torch.optim.Adam(sae.parameters(), lr=2e-3)
    for _ in range(3000):
        x = batch(512)
        loss = ((sae(x)[0] - x) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); remove_parallel_grad_(sae); opt.step(); normalize_decoder_(sae)
    cos = (D.t() @ (sae.W_dec / sae.W_dec.norm(dim=0, keepdim=True))).max(1).values
    assert (cos > 0.9).float().mean() > 0.9


# Added tests.

def test_orthonormal_dictionary_makes_the_encoder_exact():                  # Proposition 5, last claim
    torch.manual_seed(0)
    sae = SAE(16, 16, k=3, dtype=torch.float64)
    with torch.no_grad():
        Q = torch.linalg.qr(torch.randn(16, 16, **f64))[0]
        sae.W_dec.copy_(Q); sae.W_enc.copy_(Q.T)
    x = torch.randn(5, 16, **f64)
    torch.testing.assert_close(sae.encode(x), topk_activation(x @ Q, 3))     # best 3-sparse code of x in basis Q

def test_jumprelu_threshold_gets_a_gradient():
    torch.manual_seed(0)
    sae = SAE(8, 32, kind="jumprelu", dtype=torch.float64, eps=0.5)
    x = torch.randn(64, 8, **f64)
    x_hat, z = sae(x)
    (((x_hat - x) ** 2).sum(1).mean() + sae.sparsity_penalty(x, z)).backward()
    assert sae.log_theta.grad.abs().sum() > 0

def test_aux_k_loss_trains_only_dead_latents():
    from lm.interp.sae import aux_k_loss
    torch.manual_seed(0)
    sae = SAE(8, 32, k=2, dtype=torch.float64)
    x = torch.randn(64, 8, **f64)
    dead = torch.zeros(32, dtype=torch.bool); dead[:5] = True
    x_hat, _ = sae(x)
    aux_k_loss(sae, x, x_hat, dead, k_aux=4).backward()
    assert sae.W_dec.grad[:, 5:].abs().sum() == 0 and sae.W_dec.grad[:, :5].abs().sum() > 0

def test_activation_buffer_drops_position_zero_and_normalises():
    from lm.model import Config, GPT
    from lm.interp.sae import ActivationBuffer
    torch.manual_seed(0)
    model = GPT(Config(vocab_size=64, d_model=16, n_layers=2, n_heads=2, n_kv_heads=1, max_seq_len=8)).eval()
    buf = ActivationBuffer(model, "blocks.0.hook_resid_post", lambda: torch.randint(0, 63, (4, 8)), size=256,
                           eot=63, store_dtype=torch.float32, autocast=False)
    assert buf.n_tokens == 32 * 10                                           # 7 kept positions per row, 4 rows per batch
    x = torch.cat([buf.next(32) for _ in range(20)])                          # crosses two refills
    assert x.pow(2).sum(-1).mean().item() == pytest.approx(16, rel=0.3)
    assert all(len(m._forward_hooks) == 0 for m in model.modules())
