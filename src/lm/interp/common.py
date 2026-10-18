"""Shared helpers for the E1 and E2 scripts."""
import json
from pathlib import Path

import torch

from lm.model import GPT, Config


def load_model(path: str, device: str = "cuda") -> GPT:
    """Load a checkpoint in the train_a3 format: {"cfg": asdict(cfg), "model": state_dict}."""
    ck = torch.load(path, map_location=device, weights_only=True)
    model = GPT(Config(**ck["cfg"]))
    model.load_state_dict(ck["model"])
    return model.to(device).eval()


def heatmap(ax, M, title: str, xlabel: str, ylabel: str, cmap: str = "viridis", center: bool = False):
    M = torch.as_tensor(M).float().cpu()
    lim = M.abs().max().item() if center else None
    im = ax.imshow(M, cmap="RdBu_r" if center else cmap, aspect="auto", vmin=-lim if center else None, vmax=lim)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.figure.colorbar(im, ax=ax)


def save_json(obj, path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=1))
    print(json.dumps(obj, indent=1))
