import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint


def _ce_sum(h: Tensor, W: Tensor, t: Tensor, ignore_index: int) -> Tensor:
    z = F.linear(h, W)
    if z.dtype in (torch.bfloat16, torch.float16):
        z = z.float()
    return F.cross_entropy(z, t, ignore_index=ignore_index, reduction="sum")


def _reduce(total: Tensor, targets: Tensor, reduction: str, ignore_index: int) -> Tensor:
    if reduction == "sum":
        return total
    # Divide by the count over the whole input, not per chunk (ratio of sums).
    return total / (targets != ignore_index).sum()


def chunked_cross_entropy(h: Tensor, W: Tensor, targets: Tensor, chunk: int, ignore_index: int = -100,
                          reduction: str = "mean") -> Tensor:
    """Cross-entropy of h @ W.T. Only one (chunk, V) block of logits is alive at a time (C1).

    Each chunk is checkpointed. The forward keeps only its slice of h. The backward recomputes its logits.
    """
    total = sum(checkpoint(_ce_sum, h[i : i + chunk], W, targets[i : i + chunk], ignore_index, use_reentrant=False)
                for i in range(0, h.shape[0], chunk))
    return _reduce(total, targets, reduction, ignore_index)


def cross_entropy_from_hidden(h: Tensor, W: Tensor, targets: Tensor, chunk: int | None = None,
                              reduction: str = "mean", ignore_index: int = -100) -> Tensor:
    """h (N, d) normalised hidden states, W (V, d) the readout. Logits in fp32."""
    if chunk:
        return chunked_cross_entropy(h, W, targets, chunk, ignore_index, reduction)
    return _reduce(_ce_sum(h, W, targets, ignore_index), targets, reduction, ignore_index)
