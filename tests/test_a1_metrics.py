import math

import numpy as np

from lm.eval.metrics import BpbMeter, bits_per_byte, bpb_from_token_loss
from lm.baselines.ngram import NGram, sample, sharded_unique, window_keys

rng = np.random.default_rng(0)


def test_uniform_is_8_bits():
    n = 1000
    assert abs(bits_per_byte(n * math.log(256), n) - 8.0) < 1e-12


def test_conditionals_sum_to_one():
    x = rng.integers(0, 8, 5000)
    m = NGram(3, alpha=0.5).fit(x)
    for h in ([1, 2], [7, 7], [200, 3]):
        assert abs(np.exp(m.next_logprobs(h)).sum() - 1) < 1e-9


def test_two_code_paths_agree():
    x = rng.integers(0, 8, 2000)
    m = NGram(3, alpha=0.5).fit(x[:1500])
    y = x[1500:1600]
    manual = -sum(m.next_logprobs(y[t - 2 : t])[y[t]] for t in range(2, len(y)))
    assert abs(m.nll(y) - manual) < 1e-8


def test_ratio_of_sums_not_mean_of_ratios():
    meter = BpbMeter()
    meter.update(100.0, 10, 40)
    meter.update(1.0, 1, 1)
    assert abs(meter.bpb - bits_per_byte(101.0, 41)) < 1e-12
    assert abs(bpb_from_token_loss(101.0 / 11, 11, 41) - meter.bpb) < 1e-12


def test_sharded_counts_match_global():
    k = window_keys(rng.integers(0, 256, 10_000), 3)
    a, b = np.unique(k, return_counts=True)
    c, d = sharded_unique(k, 7)
    assert np.array_equal(a, c) and np.array_equal(b, d)


def test_sampler_is_deterministic_under_seed():
    x = rng.integers(97, 100, 3000)
    m = NGram(2, alpha=0.1).fit(x)
    assert sample(m, b"a", 50, seed=3) == sample(m, b"a", 50, seed=3)
