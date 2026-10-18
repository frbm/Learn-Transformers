import math


def lr_mult(step: int, steps: int, warmup: int, schedule: str = "cosine", decay_frac: float = 0.2,
            min_frac: float = 0.0, progress: float | None = None) -> float:
    """Multiplier of each group's base learning rate. Linear warm-up, then cosine to min_frac, or WSD.

    progress in [0, 1] replaces step / steps when the run has a wall-clock budget instead of a step count.
    """
    if step < warmup:
        return (step + 1) / warmup
    if schedule == "cosine":
        p = progress if progress is not None else (step - warmup) / max(1, steps - warmup)
        return min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    if schedule == "wsd":
        # Warm-up, stable, then a linear decay over the last decay_frac of the run.
        left = (1 - progress) / decay_frac if progress is not None else (steps - step) / max(1, int(steps * decay_frac))
        return min_frac + (1 - min_frac) * min(1.0, max(0.0, left))
    if schedule == "constant":
        return 1.0
    raise ValueError(f"unknown schedule {schedule!r}")
