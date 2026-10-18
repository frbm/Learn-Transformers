"""A4 Task 4: bpb against position at T = 2048 for RoPE (four scalings) and NoPE, without fine-tuning."""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lm.data.tokens import TokenData, per_position_loss
from lm.model import GPT, Config

T_TRAIN = 512


def load(path: str, T: int, scaling: dict | None) -> GPT:
    ck = torch.load(path, map_location="cuda")
    # The RoPE tables are non-persistent buffers. So max_seq_len and the scaling change at load time.
    cfg = Config(**{**ck["cfg"], "max_seq_len": T, "rope_scaling": scaling})
    m = GPT(cfg).cuda()
    m.load_state_dict(ck["model"])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rope", default="checkpoints/a4_rope.pt")
    ap.add_argument("--nope", default="checkpoints/a4_nope.pt")
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--windows", type=int, default=64)
    ap.add_argument("--bucket", type=int, default=64)
    args = ap.parse_args()

    s = args.T / T_TRAIN
    settings = {
        "RoPE": ("rope", None),
        "RoPE + PI": ("rope", {"type": "pi", "factor": s}),
        "RoPE + NTK": ("rope", {"type": "ntk", "factor": s}),
        "RoPE + YaRN": ("rope", {"type": "yarn", "factor": s, "orig_ctx": T_TRAIN, "alpha": 1, "beta": 32}),
        "NoPE": ("nope", None),
    }
    data = TokenData()
    x, y = data.val_windows(args.windows, args.T)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, (kind, scaling) in settings.items():
        m = load(args.rope if kind == "rope" else args.nope, args.T, scaling)
        loss = per_position_loss(m, x, y).cpu().numpy()
        b = loss.reshape(-1, args.bucket).mean(1)
        bpb = b / (np.log(2) * data.bytes_per_token)
        ax.plot(np.arange(len(b)) * args.bucket + args.bucket / 2, bpb, label=name)
        print(f"{name:>12}: bpb <512 {bpb[: T_TRAIN // args.bucket].mean():.3f} | >=512 {bpb[T_TRAIN // args.bucket:].mean():.3f}")
        del m
        torch.cuda.empty_cache()
    ax.axvline(T_TRAIN, color="k", ls="--", lw=1)
    ax.set(xlabel="position", ylabel="bpb", ylim=(None, None))
    ax.legend()
    fig.tight_layout()
    fig.savefig("runs/a4_extrapolation.png", dpi=150)


if __name__ == "__main__":
    main()
