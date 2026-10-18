import math
from dataclasses import asdict, dataclass, fields


def round_up(x: int, m: int) -> int:
    return -(-x // m) * m


@dataclass
class Config:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    max_seq_len: int
    mlp_multiple_of: int = 64
    norm_eps: float = 1e-6
    # A3 ablation switches.
    use_residual: bool = True
    use_mlp: bool = True
    # A4 positions.
    pos_encoding: str = "rope"  # "rope" | "nope"
    rope_base: float = 10_000.0
    rope_scaling: dict | None = None
    # "sdpa" | "naive". The naive path is the A3 oracle.
    attn_impl: str = "sdpa"
    # B1 parametrisation. Used by lm.model.init.
    parametrization: str = "sp"  # "sp" | "mup"
    base_width: int = 256
    init_std: float = 0.02
    scale_out_proj: bool = True
    # C1 memory: recompute each block in backward.
    checkpointing: bool = False

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    @property
    def mlp_hidden(self) -> int:
        return round_up(math.ceil(8 * self.d_model / 3), self.mlp_multiple_of)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Ignore unknown keys. Then old checkpoints load after a field is removed."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def expected_params(cfg: Config, tied: bool = True) -> int:
    """Derived on paper (A3 Task 1), not counted from modules."""
    d, dh, h = cfg.d_model, cfg.d_head, cfg.mlp_hidden
    attn = d * cfg.n_heads * dh + 2 * d * cfg.n_kv_heads * dh + cfg.n_heads * dh * d + 2 * dh
    mlp = 3 * d * h if cfg.use_mlp else 0
    norms = 2 * d if cfg.use_mlp else d
    emb = cfg.vocab_size * d * (1 if tied else 2)
    return emb + cfg.n_layers * (attn + mlp + norms) + d
