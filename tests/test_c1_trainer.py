import numpy as np
import pytest
import torch

from lm.model import GPT, Config

from lm.model.attention import attention_core, attention_sdpa
from lm.train.memory import static_bytes
from lm.model import expected_params
from lm.train.trainer import Trainer, TrainConfig, lr_mult

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def tiny_cfg(**kw):
    return Config(**{**dict(vocab_size=128, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=16), **kw})


def make_trainer(tmp_path, **kw):
    rng = np.random.default_rng(0)
    toks = np.tile(rng.integers(0, 128, 50), 100).astype(np.uint16)  # periodic: learnable in a few steps
    val = torch.from_numpy(rng.integers(0, 128, (4, 17)).astype(np.int64))
    tc = TrainConfig(**{**dict(name="t", B=4, T=16, K=2, steps=20, warmup=3, eval_every=1000, ckpt_every=1000,
                               device="cpu", out_dir=str(tmp_path), peak_flops=1e12), **kw})
    return Trainer(tiny_cfg(), tc, toks, (val[:, :-1], val[:, 1:]), bytes_per_token=4.0)


def test_exact_resume(tmp_path):
    full = [r["loss"] for r in make_trainer(tmp_path / "a").fit(verbose=False) if "loss" in r]
    t = make_trainer(tmp_path / "b")
    t.fit(until=10, verbose=False)
    t.save()
    t2 = make_trainer(tmp_path / "b")  # a fresh process would build exactly this object
    t2.load()
    resumed = [r["loss"] for r in t2.fit(verbose=False) if "loss" in r]
    assert resumed == full[10:]  # bitwise on CPU


def test_loss_decreases(tmp_path):
    recs = make_trainer(tmp_path, steps=60, lr=1e-2).fit(verbose=False)
    losses = [r["loss"] for r in recs if "loss" in r]
    assert np.mean(losses[-5:]) < np.mean(losses[:5]) - 0.1


def test_lr_schedule():
    assert lr_mult(0, 100, 10) == pytest.approx(0.1)
    assert lr_mult(9, 100, 10) == pytest.approx(1.0)
    assert lr_mult(100, 100, 10, min_frac=0.1) == pytest.approx(0.1)
    assert lr_mult(50, 100, 10, "wsd") == 1.0 and lr_mult(90, 100, 10, "wsd") == pytest.approx(0.5)


def test_static_bytes_is_18N():
    cfg = tiny_cfg()
    assert static_bytes(cfg) == 18 * expected_params(cfg)
    assert static_bytes(cfg, "muon") < static_bytes(cfg)


@cuda
def test_fused_attention_matches_sdpa():
    q = torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 4, 128, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(2, 4, 128, 64, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(attention_sdpa(q, k, v), attention_core(q, k, v), atol=2e-2, rtol=2e-2)


@cuda
def test_chunked_loss_matches_full_under_autocast():
    torch.manual_seed(0)
    m = GPT(tiny_cfg(max_seq_len=64)).cuda()
    idx = torch.randint(0, 128, (4, 64), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        a, b = m.loss(idx, idx), m.loss(idx, idx, ce_chunk=48)
    torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
