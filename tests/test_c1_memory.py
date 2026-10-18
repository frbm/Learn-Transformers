# tests/test_c1_memory.py
import pytest, torch, torch.nn.functional as F
from lm.model import Config, GPT
from lm.train.memory import predict_peak_memory
from lm.model.loss import chunked_cross_entropy
from lm.train.trainer import mfu

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

def tiny(**kw):
    base = dict(vocab_size=512, d_model=64, n_layers=3, n_heads=4, n_kv_heads=2, max_seq_len=64)
    return Config(**{**base, **kw})

def test_mfu_arithmetic():
    assert mfu(flops_per_step=71e12, step_time_s=2.0, peak_flops=71e12) == pytest.approx(0.5)

def test_chunked_ce_matches_full():
    torch.manual_seed(0)
    h = torch.randn(300, 32, dtype=torch.float64, requires_grad=True)
    W = torch.randn(97, 32, dtype=torch.float64, requires_grad=True)
    t = torch.randint(0, 97, (300,)); t[::7] = -100
    full = F.cross_entropy(h @ W.T, t, ignore_index=-100)
    g_full = torch.autograd.grad(full, (h, W))
    chunked = chunked_cross_entropy(h, W, t, chunk=64)         # 300 is not a multiple of 64, on purpose
    g_chunk = torch.autograd.grad(chunked, (h, W))
    torch.testing.assert_close(chunked, full)
    for a, b in zip(g_chunk, g_full): torch.testing.assert_close(a, b)

def test_grad_accumulation_is_token_weighted():
    torch.manual_seed(0)
    model = GPT(tiny()).double()
    idx = torch.randint(0, 512, (4, 32)); tgt = torch.randint(0, 512, (4, 32))
    tgt[0, :20] = -100; tgt[3, :5] = -100                        # unequal numbers of loss-bearing tokens
    model.loss(idx, tgt).backward()
    g_full = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    n = (tgt != -100).sum()
    for i in range(4):                                           # four micro-batches of one sequence
        (model.loss(idx[i:i+1], tgt[i:i+1], reduction="sum") / n).backward()
    for g, p in zip(g_full, model.parameters()): torch.testing.assert_close(p.grad, g)

def test_checkpointing_gives_same_gradients():
    torch.manual_seed(0)
    a = GPT(tiny()).double(); b = GPT(tiny(checkpointing=True)).double()
    b.load_state_dict(a.state_dict())
    idx = torch.randint(0, 512, (2, 64)); tgt = torch.randint(0, 512, (2, 64))
    a.loss(idx, tgt).backward(); b.loss(idx, tgt).backward()
    for p, q in zip(a.parameters(), b.parameters()): torch.testing.assert_close(p.grad, q.grad)

CONFIGS = [   # (shape, B, T, checkpointing, ce_chunk)
    (dict(d_model=256, n_layers=4, n_heads=4, n_kv_heads=4), 32, 512, False, None),
    (dict(d_model=512, n_layers=8, n_heads=8, n_kv_heads=4), 16, 1024, False, None),
    (dict(d_model=512, n_layers=8, n_heads=8, n_kv_heads=4), 16, 1024, True, 2048),
]

@cuda
@pytest.mark.slow
@pytest.mark.parametrize("shape,B,T,ckpt,chunk", CONFIGS)
def test_memory_predictor_within_15pct(shape, B, T, ckpt, chunk):
    cfg = Config(vocab_size=16448, max_seq_len=T, checkpointing=ckpt, **shape)
    model = GPT(cfg).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    idx = torch.randint(0, cfg.vocab_size, (B, T), device="cuda")
    for step in range(2):                                         # step 0 allocates Adam's state
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model.loss(idx, idx, ce_chunk=chunk)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        if step == 0: torch.cuda.reset_peak_memory_stats()
    actual = torch.cuda.max_memory_allocated()
    pred = predict_peak_memory(cfg, B, T, optimizer="adamw", checkpointing=ckpt, ce_chunk=chunk)
    del model, opt, loss; torch.cuda.empty_cache()
    assert abs(pred - actual) / actual < 0.15, f"predicted {pred / 2**30:.2f} GiB, measured {actual / 2**30:.2f} GiB"
