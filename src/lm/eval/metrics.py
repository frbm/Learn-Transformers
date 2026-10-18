import math

LN2 = math.log(2)


def bits_per_byte(total_nll_nats: float, n_bytes: int) -> float:
    return total_nll_nats / (LN2 * n_bytes)


def bpb_from_token_loss(mean_loss_nats: float, n_tokens: int, n_bytes: int) -> float:
    # Use totals over the full eval set. Do not average per-batch ratios.
    return mean_loss_nats * n_tokens / (LN2 * n_bytes)


def perplexity(total_nll_nats: float, n_tokens: int) -> float:
    return math.exp(total_nll_nats / n_tokens)


class BpbMeter:
    """Accumulates sums, so that the final bpb is a ratio of sums."""

    def __init__(self):
        self.nll = 0.0
        self.tokens = 0
        self.bytes = 0

    def update(self, sum_nll_nats: float, n_tokens: int, n_bytes: int) -> None:
        self.nll += float(sum_nll_nats)
        self.tokens += int(n_tokens)
        self.bytes += int(n_bytes)

    @property
    def bpb(self) -> float:
        return bits_per_byte(self.nll, self.bytes)

    @property
    def loss(self) -> float:
        return self.nll / self.tokens
