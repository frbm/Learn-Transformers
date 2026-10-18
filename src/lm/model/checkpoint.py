"""One checkpoint format for every module: {"cfg": Config.to_dict(), "model": state_dict, **extra}."""
from pathlib import Path

import torch

from lm.model.config import Config
from lm.model.gpt import GPT


def save_model(model: GPT, path: str | Path, **extra) -> None:
    """extra: anything small that must travel with the weights (special tokens, the training config)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"cfg": model.cfg.to_dict(), "model": model.state_dict(), **extra}, tmp)
    tmp.replace(path)  # Atomic: a crash during the save keeps the previous file.


def load_model(path: str | Path, device: str = "cuda", **cfg_overrides) -> tuple[GPT, dict]:
    """(model, extra). cfg_overrides change non-persistent settings (max_seq_len, rope_scaling, attn_impl)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(Config.from_dict({**ck["cfg"], **cfg_overrides}))
    model.load_state_dict(ck["model"])
    return model.to(device), {k: v for k, v in ck.items() if k not in ("cfg", "model")}
