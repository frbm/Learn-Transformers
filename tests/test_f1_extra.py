import math, pytest, torch
from lm.eval.diversity import bleu, entropy_estimate, kl, mauve, mi_estimate, pareto_front, tv
from lm.eval.embedding import _hash_embed, divpo_pair, kmeans, quantise

f64 = dict(dtype=torch.float64)


def test_bleu_partial_match_and_brevity():
    assert bleu("a b c d", ["a b c d"]) == pytest.approx(1.0)
    assert 0 < bleu("a b c d e f", ["a b c d e x"]) < 1
    assert bleu("a b c d", ["a b c d e f g h"]) == pytest.approx(math.exp(1 - 8 / 4))


def test_kmeans_recovers_separated_clusters():
    g = torch.Generator().manual_seed(0)
    centres = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]], **f64)
    X = torch.cat([c + 0.1 * torch.randn(50, 2, generator=g, **f64) for c in centres])
    lab = kmeans(X, 3)
    for b in range(3):
        assert lab[50 * b : 50 * (b + 1)].unique().numel() == 1
    assert lab.unique().numel() == 3
    hp, hq = quantise(X[:75], X[75:], 3)
    assert hp.sum().item() == pytest.approx(1.0) and hq.sum().item() == pytest.approx(1.0)


def test_mauve_of_identical_samples_is_one():
    hp, hq = quantise(torch.eye(6, **f64).repeat(3, 1), torch.eye(6, **f64).repeat(3, 1), 6)
    assert mauve(hp, hq) == pytest.approx(1.0)


def test_mi_bounds():                                                # 0 <= I <= H(C)
    g = torch.Generator().manual_seed(1)
    C, n = 4, 20_000
    prior = torch.full((C,), 0.25, **f64)
    cond = torch.multinomial(prior, n, replacement=True, generator=g)
    same = torch.zeros(n, C, **f64)                                  # q(y | c) independent of c
    assert mi_estimate(same, cond, prior.log()) == pytest.approx(0.0, abs=1e-12)
    det = torch.full((n, C), -1e9, **f64); det[torch.arange(n), cond] = 0.0   # y determines c
    assert mi_estimate(det, cond, prior.log()) == pytest.approx(math.log(C))


def test_entropy_estimate_is_ratio_of_sums():
    H, se = entropy_estimate(torch.tensor([-2.0, -6.0], **f64), torch.tensor([1, 3]))
    assert H == pytest.approx(8 / 4) and se >= 0


def test_divpo_pair_picks_far_chosen_and_close_rejected():
    E = torch.nn.functional.normalize(torch.tensor([[1, 0], [1, 0.05], [1, -0.05], [0, 1], [1, 0.01]], **f64), dim=1)
    r = torch.tensor([0.9, 0.8, 0.1, 0.95, 0.2], **f64)
    c, rej = divpo_pair(E, r, 0.5)
    assert c == 3                                                    # high reward and far from the rest
    assert rej == 4                                                  # low reward and closest to the rest
    assert divpo_pair(E, torch.ones(5, **f64), 0.5) is None


def test_pareto_front_and_tv_kl():
    assert pareto_front([(1, 1), (2, 0.5), (0.5, 2), (0.4, 0.4)]) == [2, 0, 1]
    p, q = torch.tensor([0.5, 0.5], **f64), torch.tensor([0.9, 0.1], **f64)
    assert tv(p, q) == pytest.approx(0.4)
    assert kl(p, p) == 0.0 and kl(p, q) > 0


def test_hash_embed_is_normalised_and_deterministic():
    E = _hash_embed(["the night is long", "the night is long", "a sunny day"])
    torch.testing.assert_close(E.norm(dim=1), torch.ones(3, **f64))
    assert (E[0] @ E[1]).item() == pytest.approx(1.0) and (E[0] @ E[2]).item() < 0.5
