"""B1. Maximal-update parametrisation (μP) for Adam, from the spectral condition.

Per-kind init std and Adam learning rate, with m = d_model / base_width:

| kind      | SP init                  | μP init             | SP lr | μP lr  |
|-----------|--------------------------|---------------------|-------|--------|
| embedding | init_std                 | init_std            | lr    | lr     |
| hidden    | init_std / sqrt(m')      | init_std / sqrt(m') | lr    | lr / m |
| readout   | init_std / sqrt(m')      | 0                   | lr    | lr / m |
| vector    | 1                        | 1                   | lr    | lr     |

m' = d_model / base_width in both parametrisations, so hidden init is 1/sqrt(fan-in) in SP too.
init_weights unties the readout from the embedding. A tied tensor cannot follow two rows of the table.
"""
import math

import torch
import torch.nn as nn
from torch import Tensor

from lm.model import GPT, Config, round_up

KINDS = ("embedding", "hidden", "readout", "vector")
def width_mult(cfg) -> float:
    return cfg.d_model / cfg.base_width if cfg.parametrization == "mup" else 1.0


def classify(name: str, p: Tensor) -> str:
    if p.ndim == 1:
        return "vector"
    if name.endswith("tok_emb.weight"):
        return "embedding"
    if name.endswith("lm_head.weight"):
        return "readout"
    if p.ndim == 2:
        return "hidden"
    raise ValueError(f"unclassified parameter {name} {tuple(p.shape)}")


def _is_out_proj(name: str) -> bool:
    return name.endswith(("attn.wo.weight", "mlp.w2.weight"))


@torch.no_grad()
def init_weights(model: GPT) -> None:
    """Untie the readout, then re-initialise every parameter by kind. Build the optimiser after this call."""
    cfg = model.cfg
    assert cfg.parametrization in ("sp", "mup"), cfg.parametrization
    if model.lm_head.weight is model.tok_emb.weight:
        model.lm_head.weight = nn.Parameter(torch.empty_like(model.tok_emb.weight))
    # Anchor the 1/sqrt(fan-in) init at init_std for d_model == base_width.
    fan_in_std = cfg.init_std * math.sqrt(cfg.base_width / cfg.d_model)
    out_scale = 1 / math.sqrt(2 * cfg.n_layers) if cfg.scale_out_proj else 1.0
    for name, p in model.named_parameters():
        kind = classify(name, p)
        if kind == "vector":
            p.fill_(1.0)
        elif kind == "embedding":
            p.normal_(0, cfg.init_std)
        elif kind == "readout":
            # μP readout std is O(1/m). Zero is the limit case and gives loss = log K at step 0.
            p.normal_(0, fan_in_std) if cfg.parametrization == "sp" else p.zero_()
        else:
            p.normal_(0, fan_in_std * (out_scale if _is_out_proj(name) else 1.0))


def param_groups(model: GPT, base_lr: float, weight_decay: float = 0.0) -> list[dict]:
    """One AdamW group per kind. `lr_mult` is the μP factor; the loop sets lr = schedule(step) * lr_mult.

    Weight decay holds lr * wd fixed per kind (so wd / lr_mult). Then the decay time scale is width-independent.
    Vectors get no decay.
    """
    m = width_mult(model.cfg)
    mult = {"embedding": 1.0, "hidden": 1 / m, "readout": 1 / m, "vector": 1.0}
    buckets: dict[str, list] = {k: [] for k in KINDS}
    names: dict[str, list] = {k: [] for k in KINDS}
    for name, p in model.named_parameters():
        k = classify(name, p)
        buckets[k].append(p)
        names[k].append(name)
    n_total = sum(1 for _ in model.parameters())
    assert sum(map(len, buckets.values())) == n_total, "a parameter is in no group"
    return [{"params": buckets[k], "names": names[k], "kind": k, "lr_mult": mult[k], "lr": base_lr * mult[k],
             "weight_decay": 0.0 if k == "vector" else weight_decay / mult[k]}
            for k in KINDS if buckets[k]]


def mup_config(width: int, layers: int, par: str, vocab: int, T: int, base_width: int = 256,
               scale_out_proj: bool = True) -> Config:
    """B-track model family: d_head = 64 at every width, width grows by adding heads, GQA group size 2."""
    heads = max(1, width // 64)
    return Config(vocab_size=round_up(vocab, 64), d_model=width, n_layers=layers, n_heads=heads,
                  n_kv_heads=max(1, heads // 2), max_seq_len=T, pos_encoding="rope", parametrization=par,
                  base_width=base_width, scale_out_proj=scale_out_proj)


def make_model(width: int, layers: int, par: str, vocab: int, T: int, base_width: int = 256, seed: int = 0,
               scale_out_proj: bool = True, device: str = "cuda") -> GPT:
    torch.manual_seed(seed)
    model = GPT(mup_config(width, layers, par, vocab, T, base_width, scale_out_proj)).to(device)
    init_weights(model)
    return model
