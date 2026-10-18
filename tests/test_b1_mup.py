# tests/test_b1_mup.py
import numpy as np, torch
import torch.nn.functional as F
from lm.model import Config, GPT
from lm.model.mup import classify, init_weights, param_groups

DEV = "cuda" if torch.cuda.is_available() else "cpu"
WIDTHS = [64, 128, 256, 512, 1024]                       # d_h = 64 fixed: width grows by adding heads

def make(n, par, seed=0):
    torch.manual_seed(seed)
    cfg = Config(vocab_size=1024, d_model=n, n_layers=2, n_heads=max(1, n // 64), n_kv_heads=max(1, n // 128),
                 max_seq_len=64, parametrization=par, base_width=64)
    m = GPT(cfg).to(DEV)
    init_weights(m)
    return m

def logit_change(n, par, steps=5, lr=1e-2):
    m = make(n, par)
    opt = torch.optim.AdamW(param_groups(m, lr), betas=(0.9, 0.95), weight_decay=0.0)
    x = torch.randint(0, 1024, (8, 64), generator=torch.Generator().manual_seed(1)).to(DEV)
    with torch.no_grad():
        z0 = m(x)
    for _ in range(steps):
        F.cross_entropy(m(x)[:, :-1].flatten(0, 1), x[:, 1:].flatten()).backward()
        opt.step(); opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        return (m(x) - z0).pow(2).mean().sqrt().item()        # RMS change of the logits

def slope(par):                                               # d log(change) / d log(width)
    ys = [logit_change(n, par) for n in WIDTHS]
    return np.polyfit(np.log(WIDTHS), np.log(ys), 1)[0]

def test_every_parameter_is_classified():
    kinds = {classify(name, p) for name, p in make(256, "mup").named_parameters()}
    assert kinds <= {"embedding", "hidden", "readout", "vector"} and {"embedding", "hidden"} <= kinds

def test_hidden_lr_scales_with_fan_in():                      # Proposition 3
    lrs = {g["kind"]: g["lr"] for g in param_groups(make(512, "mup"), 1e-2)}
    assert np.isclose(lrs["hidden"], 1e-2 * 64 / 512) and np.isclose(lrs["embedding"], 1e-2)

def test_mup_logit_updates_are_width_independent():
    assert abs(slope("mup")) < 0.2

def test_sp_logit_updates_grow_with_width():                  # the check must be able to fail
    assert slope("sp") > 0.3


# Added tests.

def test_sp_equals_mup_at_base_width():                       # two code paths agree at m = 1, except the readout init
    sp, mu = make(64, "sp"), make(64, "mup")
    for (n, a), b in zip(sp.named_parameters(), mu.parameters()):
        if classify(n, a) != "readout":
            torch.testing.assert_close(a, b)
    assert [g["lr"] for g in param_groups(sp, 1e-2)] == [g["lr"] for g in param_groups(mu, 1e-2)]

def test_init_std_by_kind():
    m = make(1024, "mup")
    std = {n: p.std().item() for n, p in m.named_parameters()}
    s = 0.02 * (64 / 1024) ** 0.5
    assert np.isclose(std["tok_emb.weight"], 0.02, rtol=0.05) and std["lm_head.weight"] == 0
    assert np.isclose(std["blocks.0.attn.wq.weight"], s, rtol=0.05)
    assert np.isclose(std["blocks.0.mlp.w2.weight"], s / 2, rtol=0.05)            # 1/sqrt(2L), L = 2
    assert all(p.eq(1).all() for n, p in m.named_parameters() if p.ndim == 1)

def test_groups_cover_every_parameter_once_and_decay_is_lr_invariant():
    m = make(512, "mup")
    gs = param_groups(m, 1e-2, weight_decay=0.1)
    ids = [id(p) for g in gs for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(m.parameters()))
    for g in gs:
        assert np.isclose(g["lr"] * g["weight_decay"], 0.0 if g["kind"] == "vector" else 1e-3)

def test_readout_is_untied_and_zero():
    m = make(256, "mup")
    assert m.lm_head.weight is not m.tok_emb.weight and m.lm_head.weight.eq(0).all()
    assert {classify(n, p) for n, p in m.named_parameters()} == {"embedding", "hidden", "readout", "vector"}
