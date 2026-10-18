# tests/test_d4_sampling.py
import pytest, torch
from lm.infer.sampling import temperature, top_k, top_p, min_p, typical, truncation_kl, Sampler
from lm.infer.generate import kv_cache_bytes
from lm.infer.speculative import accept_or_resample, speculative_generate, expected_tokens_per_call

f64 = dict(dtype=torch.float64)
LOGITS = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05], **f64))
kept = lambda l: torch.isfinite(l).nonzero().flatten().tolist()

def test_truncations():
    assert kept(top_k(LOGITS, 2)) == [0, 1]
    assert kept(top_p(LOGITS, 0.75)) == [0, 1] and kept(top_p(LOGITS, 0.85)) == [0, 1, 2]
    assert kept(min_p(LOGITS, 0.25)) == [0, 1, 2] and kept(min_p(LOGITS, 0.7)) == [0]

def test_top_p_unsorted_and_batched():
    out = top_p(torch.stack([LOGITS[torch.tensor([2, 0, 3, 1])], LOGITS]), 0.75)
    assert kept(out[0]) == [1, 3] and kept(out[1]) == [0, 1]

def test_temperature_and_entropy():                                 # Proposition 1
    l = torch.randn(50, **f64)
    H = lambda x: -(torch.softmax(x, -1) * torch.log_softmax(x, -1)).sum().item()
    Hs = [H(temperature(l, T)) for T in (0.3, 0.7, 1.0, 1.5, 3.0)]
    assert all(a < b for a, b in zip(Hs, Hs[1:]))
    assert torch.softmax(temperature(l, 1e-3), -1).max().item() > 0.999

def test_typical_keeps_tokens_closest_to_the_entropy():
    l = torch.randn(30, **f64) * 2
    p = torch.softmax(l, -1)
    score = (-p.log() + (p * p.log()).sum()).abs()
    k = torch.isfinite(typical(l, 0.6))
    assert p[k].sum() >= 0.6 and score[k].max() <= score[~k].min()
    assert p[k].sum() - p[score.masked_fill(~k, -1.0).argmax()] < 0.6        # minimal

def test_truncation_kl():                                           # Proposition 3
    l = torch.randn(20, **f64)
    t = top_p(l, 0.8)
    p, q = torch.softmax(l, -1), torch.softmax(t, -1)
    m = q > 0
    kl = (q[m] * (q[m].log() - p[m].log())).sum()
    torch.testing.assert_close(truncation_kl(l, t), kl)
    torch.testing.assert_close(kl, -torch.log(p[m].sum()))

def test_processor_order_matters():
    l = torch.log(torch.tensor([0.4, 0.35, 0.25], **f64))
    temp_first = Sampler([lambda x: temperature(x, 2.0), lambda x: top_p(x, 0.72)])
    trunc_first = Sampler([lambda x: top_p(x, 0.72), lambda x: temperature(x, 2.0)])
    assert kept(temp_first.process(l)) == [0, 1, 2] and kept(trunc_first.process(l)) == [0, 1]

def test_accept_or_resample():                                      # Proposition 5, one step
    g = torch.Generator().manual_seed(1)
    p, q = torch.tensor([0.5, 0.3, 0.2], **f64), torch.tensor([0.2, 0.2, 0.6], **f64)
    n, out, acc = 50_000, torch.zeros(3, **f64), 0
    for _ in range(n):
        ok, y = accept_or_resample(p, q, int(torch.multinomial(q, 1, generator=g)), g)
        out[y] += 1; acc += ok
    torch.testing.assert_close(out / n, p, atol=0.008, rtol=0)
    assert acc / n == pytest.approx(1 - 0.5 * (p - q).abs().sum().item(), abs=0.008)

def test_speculative_generate_is_exact():                           # Proposition 5, full algorithm
    g = torch.Generator().manual_seed(0)
    V = 5
    P = torch.softmax(1.5 * torch.randn(V, V, generator=g, **f64), -1)   # target bigram table
    Q = torch.softmax(1.5 * torch.randn(V, V, generator=g, **f64), -1)   # draft bigram table
    bigram = lambda T: (lambda prefixes: T[torch.tensor([s[-1] for s in prefixes])])
    n, counts, calls = 20_000, torch.zeros(V, V, **f64), 0
    for _ in range(n):
        out, stats = speculative_generate(bigram(P), bigram(Q), [0], n_tokens=2, gamma=3, generator=g)
        counts[out[0], out[1]] += 1; calls += stats["target_calls"]
    expected = n * P[0][:, None] * P
    assert ((counts - expected) ** 2 / expected).sum().item() < 51.18      # chi-square, 24 dof, p = 0.001
    assert calls < 2 * n                                                    # usually one target call per sample

def test_arithmetic():
    assert expected_tokens_per_call(0.0, 4) == pytest.approx(1.0)
    assert expected_tokens_per_call(0.8, 4) == pytest.approx((1 - 0.8 ** 5) / 0.2)
    assert expected_tokens_per_call(1.0, 4) == pytest.approx(5.0)
    assert kv_cache_bytes(32, 8, 128, 1) == 131_072                          # Llama-3.1-8B, bf16, per token
