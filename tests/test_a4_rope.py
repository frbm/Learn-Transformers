import math

import torch

from lm.model import GPT, Config
from lm.model.rope import apply_rope, rope_angles, rope_freqs

DEV = "cuda" if torch.cuda.is_available() else "cpu"
D, BASE = 64, 10_000.0


def at(x, m, cos, sin):
    return apply_rope(x.view(1, 1, 1, -1), cos, sin, offset=m)[0, 0, 0]


def test_relative_identity():
    cos, sin = rope_angles(D, 4096, BASE)
    q, k = torch.randn(D), torch.randn(D)
    for m, n, c in [(0, 5, 17), (3, 100, 900), (1000, 10, 2000)]:
        torch.testing.assert_close(at(q, m, cos, sin) @ at(k, n, cos, sin),
                                   at(q, m + c, cos, sin) @ at(k, n + c, cos, sin), atol=1e-3, rtol=1e-4)


def test_norm_preserved():
    cos, sin = rope_angles(D, 4096, BASE)
    x = torch.randn(2, 4, 50, D)
    torch.testing.assert_close(apply_rope(x, cos, sin, offset=7).norm(dim=-1), x.norm(dim=-1))


def test_angle_table_precision():
    cos, _ = rope_angles(D, 131_072, BASE)
    m = 131_071
    theta = BASE ** (-torch.arange(0, D, 2, dtype=torch.float64) / D)
    torch.testing.assert_close(cos[m].double(), torch.cos(m * theta), atol=1e-6, rtol=0)


def test_pi_and_ntk_endpoints():
    f0 = rope_freqs(D, BASE)
    torch.testing.assert_close(rope_freqs(D, BASE, {"type": "pi", "factor": 4.0}), f0 / 4)
    ntk = rope_freqs(D, BASE, {"type": "ntk", "factor": 4.0})
    assert math.isclose(ntk[0] / f0[0], 1.0) and math.isclose(ntk[-1] / f0[-1], 0.25)


def test_yarn_by_parts():
    cfg = {"type": "yarn", "factor": 4.0, "orig_ctx": 512, "alpha": 1, "beta": 32}
    f0, f = rope_freqs(D, BASE), rope_freqs(D, BASE, cfg)
    turns = 512 * f0 / (2 * math.pi)
    torch.testing.assert_close(f[turns > 32], f0[turns > 32])
    torch.testing.assert_close(f[turns < 1], f0[turns < 1] / 4)
    torch.testing.assert_close(rope_freqs(D, BASE, {**cfg, "factor": 1.0}), f0)


def test_kv_cache_with_rope():
    cfg = Config(vocab_size=1024, d_model=256, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=128, pos_encoding="rope")
    m = GPT(cfg).to(DEV).eval()
    x = torch.randint(0, 1024, (2, 40), device=DEV)
    with torch.no_grad():
        full = m(x)
        cache = m.init_cache(batch_size=2)
        parts = [m(x[:, :16], cache=cache), m(x[:, 16:24], cache=cache)]
        parts += [m(x[:, t:t + 1], cache=cache) for t in range(24, 40)]
    torch.testing.assert_close(torch.cat(parts, 1), full, atol=1e-4, rtol=1e-4)


def test_rope_buffers_not_in_checkpoint():
    cfg = Config(vocab_size=128, d_model=64, n_layers=1, n_heads=2, n_kv_heads=1, max_seq_len=32)
    sd = GPT(cfg).state_dict()
    assert not any("rope" in k for k in sd)
    long = GPT(Config(**{**cfg.__dict__, "max_seq_len": 128, "rope_scaling": {"type": "pi", "factor": 4.0}}))
    long.load_state_dict(sd)
