import numpy as np, torch
from torch.utils.flop_counter import FlopCounterMode
from lm.model import Config, GPT
from lm.analysis.scaling import flops_per_token, compute_optimal, fit_scaling_law, bootstrap

TRUE = dict(E=1.7, A=400.0, B=410.0, alpha=0.34, beta=0.28)
def law(N, D, E, A, B, alpha, beta): return E + A * N ** -alpha + B * D ** -beta

def synthetic(noise, seed=0):
    rng = np.random.default_rng(seed)
    N = np.repeat(np.logspace(6, 8, 5), 4)
    D = np.tile(np.logspace(8, 10, 4), 5)
    return N, D, law(N, D, **TRUE) * np.exp(noise * rng.standard_normal(N.size))

def test_flop_formula_matches_counter():                    # §1, forward pass, naive attention (n_ctx = T)
    cfg = Config(vocab_size=2048, d_model=256, n_layers=4, n_heads=4, n_kv_heads=2, max_seq_len=256, attn_impl="naive")
    m, x = GPT(cfg).eval(), torch.randint(0, 2048, (2, 256))
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        m(x)
    assert abs(flops_per_token(cfg, n_ctx=256, training=False) / (fc.get_total_flops() / x.numel()) - 1) < 0.05

def test_compute_optimal_matches_brute_force():             # Proposition 2
    C = 1e18
    N_star, D_star = compute_optimal(C, TRUE["A"], TRUE["B"], TRUE["alpha"], TRUE["beta"])
    Ns = np.logspace(6, 11, 20_001)
    best = Ns[np.argmin(law(Ns, C / (6 * Ns), **TRUE))]
    assert abs(np.log(N_star / best)) < 1e-3 and np.isclose(6 * N_star * D_star, C)

def test_fit_recovers_exponents():
    p = fit_scaling_law(*synthetic(noise=0.003))
    assert abs(p["alpha"] - TRUE["alpha"]) < 0.05 and abs(p["beta"] - TRUE["beta"]) < 0.05

def test_bootstrap_interval_is_honest():                    # covers the truth, and is not implausibly narrow
    lo, hi = bootstrap(*synthetic(noise=0.01), n_boot=200, level=0.99)["alpha"]
    assert lo < TRUE["alpha"] < hi and hi - lo > 0.01



def test_bootstrap_survives_a_boundary_fit():              # regression: E -> 0 collapsed every interval to a point
    # Six real smoke-sweep runs (C = 1e13, 3e13). Their best fit sits at E = 0, where exp(e) underflows.
    N = np.array([2885248, 4720832, 7146240] * 2, dtype=float)
    D = np.array([552960, 339968, 225280, 1658880, 1017856, 675840], dtype=float)
    L = np.array([6.985894680023193, 7.317311763763428, 7.44584846496582, 6.2570273876190186, 6.545027256011963,
                  6.7805376052856445])
    assert fit_scaling_law(N, D, L)["E"] == 0.0                # exp(e) has underflowed
    lo, hi = bootstrap(N, D, L, n_boot=20)["a"]
    assert np.isfinite([lo, hi]).all() and hi - lo > 0.1
