"""Shared loaders for the D4 and F1 scripts: models, tokenizer, verse headers, reward, the line-count constraint.

Every loader has a fallback (tiny random model, plain-text header, stand-in reward). The scripts then run
as smoke tests before the D1 / D3 artefacts exist.
"""
import dataclasses
import importlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from lm.tokenizer.bpe import Tokenizer
from lm.model import GPT, Config

SPECIALS = ["<|author|>", "<|title|>", "<|lines|>", "<|body|>"]
TIER1 = dict(d_model=512, n_layers=8, n_heads=8, n_kv_heads=4)      # about 33 M parameters at V = 16 384
TIER0 = dict(d_model=256, n_layers=4, n_heads=4, n_kv_heads=2)


def load_model(path: str | None, vocab: int, device="cuda", dtype=torch.bfloat16, shape: dict = TIER1,
               max_seq_len: int = 2048, seed: int = 0) -> GPT:
    """A checkpoint {"cfg": dict, "model": state_dict}, or a random model of the given shape if path is None."""
    if path:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        names = {f.name for f in dataclasses.fields(Config)}
        cfg = Config(**{k: v for k, v in ck["cfg"].items() if k in names})
        model = GPT(cfg)
        model.load_state_dict(ck["model"], strict=False)
    else:
        torch.manual_seed(seed)
        model = GPT(Config(vocab_size=-(-vocab // 64) * 64, max_seq_len=max_seq_len, **shape))
    return model.to(device=device, dtype=dtype).eval()


def load_tokenizer(path: str = "data/fineweb_tok.json") -> Tokenizer:
    return Tokenizer.load(path)


def has_specials(tok: Tokenizer) -> bool:
    return all(s in tok.special for s in SPECIALS)


def header_text(meta: dict, n_lines: int, tok: Tokenizer, instruction: str = "") -> str:
    a, t = meta["author"], meta["title"] + (f" ({instruction})" if instruction else "")
    if has_specials(tok):
        return f"<|author|> {a} <|title|> {t} <|lines|> {n_lines} <|body|>"
    return f"Author: {a}\nTitle: {t}\nLines: {n_lines}\n\n"


def count_lines(text: str) -> int:
    return sum(1 for l in text.splitlines() if l.strip())


def verse_split(split: str = "val", n: int = 200, seed: int = 0) -> list[tuple[dict, str]]:
    """(meta, poem) pairs from D1's verse loader, else synthetic stand-ins."""
    try:
        from lm.verse.corpus import load_verse
        poems = load_verse()[split]
    except Exception:
        rng = np.random.default_rng(seed)
        words = "night sea wind light heart grave rose winter field star sorrow".split()
        poems = [({"author": f"Author {i % 17}", "title": f"Poem {i}"},
                  "\n".join(" ".join(rng.choice(words, 6)) for _ in range(int(rng.integers(4, 21)))))
                 for i in range(4 * n)]
    order = np.random.default_rng(seed).permutation(len(poems))[:n]
    return [poems[i] for i in order]


def stand_in_reward(meta: dict, poem: str) -> float:
    """Stand-in for D3's programmatic reward, in [0, 1]: line-count match and lexical variety."""
    n_req, n = int(meta["lines"]), count_lines(poem)
    words = poem.split()
    variety = len(set(words)) / len(words) if words else 0.0
    return 0.5 * float(np.exp(-abs(n - n_req) / max(1.0, n_req / 4))) + 0.5 * variety


def load_reward(spec: str | None):
    """'module:function' with signature f(meta, poem) -> float. None gives the stand-in."""
    if not spec:
        return stand_in_reward
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)


def decode(tok: Tokenizer, ids: list[int]) -> str:
    return tok.decode([i for i in ids if i < tok.vocab_size]).decode("utf-8", errors="replace")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l]


class LineConstraint:
    """Token masking for "exactly n non-empty lines", vectorised over the batch with per-token tables.

    State per row: c = completed non-empty lines, f = the current line has a non-space character.
    Rules: eos is masked while c + f < n; a token that makes c + f > n is masked; eos is forced once c == n.
    """

    def __init__(self, tok: Tokenizer, V: int, n: Tensor, eos_id: int):
        dev = n.device
        nl, first, mid, last, nonws = (np.zeros(V, dtype=np.int64) for _ in range(5))
        for v, b in enumerate(tok.vocab[:V]):
            if v == eos_id or b.startswith(b"<|"):
                continue
            segs = b.split(b"\n")
            nl[v] = len(segs) - 1
            nonws[v] = bool(b.strip())
            first[v] = bool(segs[0].strip())
            mid[v] = sum(bool(s.strip()) for s in segs[1:-1])
            last[v] = bool(segs[-1].strip())
        t = lambda a: torch.from_numpy(a).to(dev)
        self.nl, self.first, self.mid, self.last, self.nonws = map(t, (nl, first, mid, last, nonws))
        self.n, self.eos = n, eos_id
        self.c = torch.zeros_like(n)
        self.f = torch.zeros_like(n)

    def _after(self):
        """(c', f') for every row and every token, shape (B, V)."""
        has_nl = self.nl[None] > 0
        f_first = self.f[:, None] | self.first[None]
        c_new = torch.where(has_nl, self.c[:, None] + f_first + self.mid[None], self.c[:, None])
        f_new = torch.where(has_nl, self.last[None].expand_as(c_new), self.f[:, None] | self.nonws[None])
        return c_new, f_new

    def apply(self, logits: Tensor) -> Tensor:
        c_new, f_new = self._after()
        bad = c_new + f_new > self.n[:, None]
        bad[:, self.eos] = (self.c + self.f) < self.n
        force = self.c >= self.n
        bad[force] = True
        bad[force, self.eos] = False
        return logits.masked_fill(bad, float("-inf"))

    def step(self, tokens: Tensor) -> None:
        c_new, f_new = self._after()
        self.c = c_new.gather(1, tokens[:, None])[:, 0]
        self.f = f_new.gather(1, tokens[:, None])[:, 0]


class steer:
    """Context manager: add alpha * vec to the residual stream after block `layer`.

    The hook sits on the block's MLP. The engine calls the submodules, not Block.forward, so a hook on the
    MLP output is the one that fires on both code paths. x + mlp(.) + alpha v is the same residual update.
    """

    def __init__(self, model, layer: int, vec: Tensor, alpha: float):
        self.mod, self.add = model.blocks[layer].mlp, (alpha * vec).to(model.tok_emb.weight)

    def __enter__(self):
        self.h = self.mod.register_forward_hook(lambda m, i, o: o + self.add)
        return self

    def __exit__(self, *exc):
        self.h.remove()
