# tests/test_d2_lora.py
import pytest, torch, torch.nn as nn
from lm.model import Config, GPT
from lm.model.lora import LoRALinear, apply_lora, lora_state_dict
from lm.model.quant import (nf4_codebook, quantize_blockwise, dequantize_blockwise, double_quantize,
                   dequantize_absmax, bits_per_param, NF4Linear)

NF4 = [-1.0, -0.6961928, -0.5250731, -0.3949175, -0.2844414, -0.1847734, -0.0910500, 0.0,
       0.0795803, 0.1609302, 0.2461123, 0.3379152, 0.4407098, 0.5626170, 0.7229568, 1.0]

def lin(d_in=48, d_out=32, seed=0):
    torch.manual_seed(seed)
    return nn.Linear(d_in, d_out, bias=False).double()

@pytest.mark.parametrize("dora", [False, True])
def test_identity_at_init(dora):
    base, x = lin(), torch.randn(5, 48, dtype=torch.float64)
    torch.testing.assert_close(LoRALinear(base, r=4, alpha=8, dora=dora).double()(x), base(x))

@pytest.mark.parametrize("scaling,expected", [("standard", 8 / 4), ("rs", 8 / 2)])
def test_scaling(scaling, expected):
    assert LoRALinear(lin(), r=4, alpha=8, scaling=scaling).scale == pytest.approx(expected)

@pytest.mark.parametrize("dora", [False, True])
def test_merge_matches_forward(dora):
    m = LoRALinear(lin(), r=4, alpha=8, dora=dora).double()
    with torch.no_grad():
        m.lora_B.normal_()
        if dora: m.lora_magnitude.mul_(1.5)
    x = torch.randn(5, 48, dtype=torch.float64)
    torch.testing.assert_close(m.merge()(x), m(x))

def test_first_sgd_step_is_projected_gradient():             # Proposition 3
    base = lin(); m = LoRALinear(base, r=4, alpha=8).double()
    x, y = torch.randn(64, 48, dtype=torch.float64), torch.randn(64, 32, dtype=torch.float64)
    W = base.weight.detach().clone().requires_grad_(True)
    G, = torch.autograd.grad(((x @ W.T - y) ** 2).mean(), W)
    ((m(x) - y) ** 2).mean().backward()
    eta = 0.1
    with torch.no_grad():
        dW = m.scale * (m.lora_B - eta * m.lora_B.grad) @ (m.lora_A - eta * m.lora_A.grad)
    torch.testing.assert_close(dW, -eta * m.scale ** 2 * G @ m.lora_A.T @ m.lora_A)

def test_pissa_init():
    base = lin(); W0 = base.weight.detach().clone()
    m = LoRALinear(base, r=4, alpha=4, init="pissa").double()
    torch.testing.assert_close(m.base.weight + m.scale * m.lora_B @ m.lora_A, W0)
    assert torch.linalg.matrix_norm(m.base.weight, ord=2).item() == pytest.approx(torch.linalg.svdvals(W0)[4].item())

def test_apply_lora_freezes_everything_else():
    cfg = Config(vocab_size=256, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=32)
    model = apply_lora(GPT(cfg), targets=["attn", "mlp"], r=4, alpha=8)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n for n in trainable)
    assert set(lora_state_dict(model)) == trainable

def test_nf4_codebook():
    torch.testing.assert_close(nf4_codebook(), torch.tensor(NF4), atol=1e-6, rtol=0)

def test_blockwise_roundtrip_and_storage():
    torch.manual_seed(0)
    w, code = torch.randn(256, 128) * 0.02, nf4_codebook()
    packed, absmax = quantize_blockwise(w, code, blocksize=64)
    assert packed.dtype == torch.uint8 and packed.numel() == w.numel() // 2 and absmax.shape == (w.numel() // 64,)
    w_hat = dequantize_blockwise(packed, absmax, code, w.shape, blocksize=64, dtype=torch.float32)
    err = (w_hat - w).abs().reshape(-1, 64) / absmax[:, None]
    assert err.max() <= (code[1:] - code[:-1]).max() / 2 + 1e-6      # nearest-point rounding
    assert torch.equal(w_hat.reshape(-1, 64).abs().amax(1), absmax)     # the block maximum is exact

def test_double_quantization():
    torch.manual_seed(0)
    absmax = torch.rand(4096) * 0.1 + 0.01
    q, scale2, offset = double_quantize(absmax, blocksize=256)
    assert q.dtype == torch.uint8 and scale2.numel() == 4096 // 256
    assert ((dequantize_absmax(q, scale2, offset, blocksize=256) - absmax).abs() / absmax).max() < 0.05

def test_bits_per_param():
    assert bits_per_param(double_quant=False) == pytest.approx(4.5)
    assert bits_per_param() == pytest.approx(4 + 8 / 64 + 32 / (64 * 256))

def test_nf4_linear():
    torch.manual_seed(0)
    base = nn.Linear(128, 64, bias=False)
    q, x = NF4Linear(base), torch.randn(3, 128)
    W_hat = q.dequantized_weight(torch.float32)
    torch.testing.assert_close(q(x), x @ W_hat.T)
    assert (W_hat - base.weight).norm() / base.weight.norm() < 0.15
    torch.testing.assert_close(LoRALinear(q, r=4, alpha=8)(x), q(x))   # LoRA on a quantised base
