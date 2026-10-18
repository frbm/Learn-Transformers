# tests/test_d3_prefs.py
import math, pytest, torch
from lm.rl.prefs import dpo_loss, kl_estimators, group_advantages, policy_loss, bon_kl

f64 = dict(dtype=torch.float64)

def test_dpo_loss_value_metrics_and_gradient():
    z = torch.zeros(4, **f64)
    assert dpo_loss(z, z, z, z, beta=0.1)[0].item() == pytest.approx(math.log(2))
    pi_w = torch.tensor([-10.0], **f64, requires_grad=True)
    pi_l = torch.tensor([-12.0], **f64, requires_grad=True)
    ref = torch.tensor([-11.0], **f64)
    loss, m = dpo_loss(pi_w, pi_l, ref, ref, beta=0.5)
    assert m["margin"].item() == pytest.approx(1.0)
    assert loss.item() == pytest.approx(-math.log(1 / (1 + math.exp(-1.0))))
    loss.backward()
    assert pi_w.grad.item() < 0 < pi_l.grad.item()

def test_dpo_population_minimiser_is_the_gibbs_tilt():          # Proposition 2(c)
    torch.manual_seed(0)
    K, beta = 6, 0.5
    log_ref = torch.log_softmax(torch.randn(K, **f64), 0)
    r = torch.randn(K, **f64)
    I, J = torch.meshgrid(torch.arange(K), torch.arange(K), indexing="ij")
    i, j = I[I != J], J[I != J]                                      # all ordered pairs, i as the winner
    p = torch.sigmoid(r[i] - r[j])                                    # Bradley-Terry probability that i wins
    theta = log_ref.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([theta], max_iter=500, tolerance_grad=1e-12, tolerance_change=1e-14,
                            line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        logp = torch.log_softmax(theta, 0)
        loss = (p * dpo_loss(logp[i], logp[j], log_ref[i], log_ref[j], beta, reduction="none")[0]).sum()
        loss.backward()
        return loss
    opt.step(closure)
    torch.testing.assert_close(torch.log_softmax(theta, 0), torch.log_softmax(log_ref + r / beta, 0),
                               atol=1e-5, rtol=0)

def test_kl_estimators():                                        # Proposition 5
    torch.manual_seed(0)
    lp, lq = torch.log_softmax(torch.randn(10, **f64), 0), torch.log_softmax(torch.randn(10, **f64), 0)
    kl = (lp.exp() * (lp - lq)).sum()
    y = torch.multinomial(lp.exp(), 200_000, replacement=True)
    k = kl_estimators(lp[y], lq[y])
    assert (k["k3"] >= 0).all() and (k["k1"] < 0).any()
    for name in ("k1", "k3"):
        assert k[name].mean().item() == pytest.approx(kl.item(), abs=4 * k[name].std().item() / math.sqrt(len(y)))

def test_group_advantages():                                     # Proposition 4
    R = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.5, 0.5, 0.5, 0.5], [3.0, 1.0, 2.0, 0.0]], **f64)
    A, G = group_advantages(R, scale="none"), R.shape[1]
    torch.testing.assert_close(A.sum(1), torch.zeros(3, **f64))
    loo = (R.sum(1, keepdim=True) - R) / (G - 1)
    torch.testing.assert_close(A, (G - 1) / G * (R - loo))
    As = group_advantages(R, scale="std")
    assert torch.isfinite(As).all() and (As[1] == 0).all()

def test_policy_loss_aggregation_and_clipping():
    mask = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]], **f64)
    adv, old = torch.tensor([1.0, -1.0], **f64), torch.zeros(2, 4, **f64)
    new = old.clone().requires_grad_(True)
    assert policy_loss(new, old, adv, mask, aggregation="sequence").item() == pytest.approx(0.0)
    assert policy_loss(new, old, adv, mask, aggregation="token").item() == pytest.approx(-0.5)
    assert policy_loss(new, old, adv, mask, aggregation="constant", max_len=4).item() == pytest.approx(-0.25)
    assert policy_loss(new, old, adv, mask, aggregation="token", logp_ref=old, beta=1.0).item() == pytest.approx(-0.5)
    new = torch.full((2, 4), 0.5, **f64).requires_grad_(True)           # ratio e^0.5 ≈ 1.65
    policy_loss(new, old, adv, mask, eps_low=0.2, eps_high=0.28, aggregation="token").backward()
    assert torch.all(new.grad[0, :3] == 0)                                 # A > 0, ratio > 1 + eps_high: clipped
    assert new.grad[1, 0] != 0                                             # A < 0: the unclipped term is the min

def test_bon_kl():                                               # Proposition 6
    assert bon_kl(1) == 0.0 and bon_kl(16) == pytest.approx(math.log(16) - 15 / 16)

def test_verse_reward_resists_the_obvious_hack():
    pytest.importorskip("pronouncing")
    from lm.verse.rewards import rhymes, verse_reward
    assert rhymes("cat", "hat") and not rhymes("cat", "cat") and not rhymes("cat", "dog")
    honest = "The cat sat on the mat\nAnd wore a little hat\nThe dog ran through the fog\nAnd jumped upon a log"
    assert verse_reward(4, honest)["total"] > verse_reward(4, "The cat\nThe cat\nThe cat\nThe cat")["total"]
