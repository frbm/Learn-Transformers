# tests/test_f1_diversity.py
import pytest, torch
from lm.eval.diversity import distinct_n, self_bleu, vendi, divergence_curve, mauve, mi_estimate, tilt

f64 = dict(dtype=torch.float64)

def test_distinct_n():
    assert distinct_n(["a b a b"], 1) == pytest.approx(2 / 4)
    assert distinct_n(["a b a b"], 2) == pytest.approx(2 / 3)
    assert distinct_n(["a b c", "a b d"], 2) == pytest.approx(3 / 4)

def test_self_bleu_extremes():
    same = ["the night is long and cold"] * 3
    assert self_bleu(same) == pytest.approx(1.0)
    assert self_bleu(["a b c d e", "f g h i j", "k l m n o"]) == 0.0

def test_vendi_counts_effective_distinct_items():                     # Proposition 2
    n = 12
    assert vendi(torch.eye(n, **f64)) == pytest.approx(n)
    assert vendi(torch.ones(n, n, **f64)) == pytest.approx(1.0)
    blocks = torch.block_diag(*[torch.ones(4, 4, **f64)] * 3)            # 3 groups of 4 identical items
    assert vendi(blocks) == pytest.approx(3.0)
    g = torch.Generator().manual_seed(0)
    X = torch.nn.functional.normalize(torch.randn(n, 5, generator=g, **f64), dim=1)
    K = X @ X.T
    assert 1.0 <= vendi(K) <= n
    Xd = X.repeat_interleave(2, 0)                                         # duplicating every item changes nothing
    assert vendi(Xd @ Xd.T) == pytest.approx(vendi(K), rel=1e-9)

def test_mauve_extremes_and_monotonicity():                           # Proposition 3
    p = torch.tensor([0.4, 0.3, 0.2, 0.1, 0.0, 0.0], **f64)
    q_far = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.5, 0.5], **f64)
    assert mauve(p, p) == pytest.approx(1.0, abs=1e-9)
    assert mauve(p, q_far) < 0.1
    scores = [mauve(p, a * p + (1 - a) * q_far) for a in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(s1 < s2 for s1, s2 in zip(scores, scores[1:]))
    for x, y in divergence_curve(p, q_far, [0.1, 0.5, 0.9]):           # mixtures keep every KL finite
        assert 0.0 < x <= 1.0 and 0.0 < y <= 1.0

def test_mutual_information_estimator():                               # Proposition 7
    g = torch.Generator().manual_seed(0)
    C, V, n = 3, 6, 50_000
    prior = torch.tensor([0.5, 0.3, 0.2], **f64)
    q = torch.softmax(2 * torch.randn(C, V, generator=g, **f64), -1)   # q(y | c)
    marg = prior @ q
    exact = sum(prior[c] * (q[c] * (q[c] / marg).log()).sum() for c in range(C)).item()
    cond = torch.multinomial(prior, n, replacement=True, generator=g)
    y = torch.multinomial(q[cond], 1, generator=g).squeeze(1)
    logq = q[:, y].T.log()                                               # (n, C): log q(y_i | c) for every c
    assert mi_estimate(logq, cond, prior.log()) == pytest.approx(exact, abs=0.02)

def test_tilting_increases_kl_not_necessarily_entropy():              # Proposition 5
    g = torch.Generator().manual_seed(0)
    log_ref = torch.log_softmax(3 * torch.randn(50, generator=g, **f64), 0)
    r = torch.randn(50, generator=g, **f64)
    KL = lambda t: (tilt(log_ref, r, t).exp() * (tilt(log_ref, r, t) - log_ref)).sum().item()
    for t in (0.3, 1.0, 2.0):
        p = tilt(log_ref, r, t).exp()
        var = (p * r ** 2).sum() - (p * r).sum() ** 2
        assert (KL(t + 1e-5) - KL(t - 1e-5)) / 2e-5 == pytest.approx(t * var.item(), rel=1e-4)
    H = lambda t: -(tilt(log_ref, r, t).exp() * tilt(log_ref, r, t)).sum().item()
    flat_r = -log_ref                                                   # a reward that prefers rare outputs
    assert H(0.0) < -(tilt(log_ref, flat_r, 0.5).exp() * tilt(log_ref, flat_r, 0.5)).sum().item()

def test_typicality_bias_is_an_implicit_temperature():                # Proposition 6
    g = torch.Generator().manual_seed(1)
    log_ref = torch.log_softmax(torch.randn(40, generator=g, **f64), 0)
    r_true = torch.randn(40, generator=g, **f64)
    alpha, beta = 0.5, 0.25
    lhs = tilt(log_ref, alpha * log_ref + r_true, 1 / beta)
    rhs = torch.log_softmax((1 + alpha / beta) * log_ref + r_true / beta, 0)
    torch.testing.assert_close(lhs, rhs)
