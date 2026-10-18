"""Trainer callbacks. A callback has before_step(trainer) and after_step(trainer); both are optional.

Callbacks run outside the timed region of a step. A probe must not tax the optimiser that it measures.
"""
import torch

from lm.model.mup import classify
from lm.train.optim import spectral_norm_estimate


class Callback:
    def before_step(self, trainer) -> None: ...

    def after_step(self, trainer) -> None: ...


class UpdateTracker(Callback):
    """B2 Task 4: ||dW||_2 / ||W||_2 per hidden matrix every `every` steps. Power iteration, no SVD."""

    def __init__(self, every: int):
        self.every, self.records = every, []
        self._before: dict[str, torch.Tensor] | None = None
        self._vecs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}  # warm starts for the power iterations

    def _hidden(self, model):
        return {n: p for n, p in model.named_parameters() if classify(n, p) == "hidden"}

    def before_step(self, trainer) -> None:
        if trainer.step % self.every == 0:
            self._before = {n: p.detach().clone() for n, p in self._hidden(trainer.model).items()}

    def after_step(self, trainer) -> None:
        if self._before is None:
            return
        rec = {"step": trainer.step - 1}
        for n, p in self._hidden(trainer.model).items():
            if n not in self._vecs:
                self._vecs[n] = (torch.randn(p.shape[1], 8, device=p.device), torch.randn(p.shape[1], 8, device=p.device))
            dw, w = self._vecs[n]
            rec[n] = spectral_norm_estimate(p - self._before[n], v=dw) / spectral_norm_estimate(p, v=w)
        self.records.append(rec)
        self._before = None
