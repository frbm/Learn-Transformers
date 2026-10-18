"""D3 shared parts: policy loading, headers, batched sampling with a KV cache, per-token statistics, scoring.

The D1 checkpoint is {"cfg": dict, "model": state_dict, "special": [names]} (experiments/d1/run.py). Without one,
`--random-init` builds a tiny random GPT and plain-text headers. Then every script runs as a smoke test.
"""
import copy
import dataclasses
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import Tensor

from lm.tokenizer.bpe import Tokenizer
from lm.model import GPT, Config
from lm.rl.prefs import completion_logits, group_advantages, kl_estimators, policy_loss, readout
from lm.verse.rewards import naive_rhyme_reward, verse_reward

SPECIALS = ["<|author|>", "<|title|>", "<|lines|>", "<|body|>"]
EOT = "<|endoftext|>"
TINY = dict(d_model=64, n_layers=2, n_heads=4, n_kv_heads=2)
ROOT = Path.cwd()


def load_policy(ckpt: str | None, tok_path: str = str(ROOT / "data/fineweb_tok.json"), device: str = "cuda",
                shape: dict = TINY, seed: int = 0) -> tuple[GPT, Tokenizer]:
    base = Tokenizer.load(tok_path)
    if ckpt:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        tok = Tokenizer(base.merges, ck.get("special", list(base.special)))
        names = {f.name for f in dataclasses.fields(Config)}
        model = GPT(Config(**{k: v for k, v in ck["cfg"].items() if k in names}))
        model.load_state_dict(ck["model"])
    else:
        torch.manual_seed(seed)
        tok = base
        model = GPT(Config(vocab_size=-(-tok.vocab_size // 64) * 64, max_seq_len=512, **shape))
    return model.to(device).eval(), tok


def frozen_copy(model: GPT) -> GPT:
    ref = copy.deepcopy(model).eval()
    ref.requires_grad_(False)
    return ref


def header_text(h: dict, tok: Tokenizer) -> str:
    if all(s in tok.special for s in SPECIALS):
        return f"<|author|> {h['author']} <|title|> {h['title']} <|lines|> {h['lines']} <|body|>"
    return f"Author: {h['author']}\nTitle: {h['title']}\nLines: {h['lines']}\n\n"


def load_headers(split: str, n: int, seed: int = 0) -> list[dict]:
    """Headers (author, title, requested lines) from D1's verse corpus. Synthetic ones if it is missing."""
    rng = random.Random(f"{split}-{seed}")
    try:
        from lm.verse.corpus import load_verse
        poems = load_verse(str(ROOT / "data/verse.jsonl"), str(ROOT / "data/corpus.txt"))[split]
        rows = [dict(author=m["author"], title=m["title"], lines=sum(1 for l in t.splitlines() if l.strip()))
                for m, t in poems]
    except Exception:
        authors = ["Emily Dickinson", "William Blake", "John Keats", "Christina Rossetti", "Walt Whitman"]
        words = ["Night", "Sea", "Rose", "Winter", "Song", "Field", "Bell", "River", "Stone", "Lamp"]
        rows = [dict(author=rng.choice(authors), title=f"The {rng.choice(words)} of {rng.choice(words)}",
                     lines=rng.choice([4, 6, 8, 12, 14, 20])) for _ in range(4 * n)]
    rng.shuffle(rows)
    return rows[:n]


def autocast(device) -> object:
    return torch.autocast("cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") else nullcontext()


@torch.no_grad()
def sample(model: GPT, prompts: list[list[int]], max_new: int, eot: int, temperature: float = 1.0,
           gen: torch.Generator | None = None, return_logp: bool = False):
    """Batched ancestral sampling with the KV cache, for prompts of different lengths.

    All rows start at the shortest prompt. A row still inside its prompt takes its next prompt token
    (teacher forcing) instead of the sample. So one batch holds every prompt, with no left padding.
    Sample from exactly softmax(z / T) in fp32: the GRPO ratios use these log-probabilities.
    With return_logp, also return the sampler's log-probabilities of its tokens (the KV-cache path).
    """
    dev = model.tok_emb.weight.device
    B = len(prompts)
    lp = torch.tensor([len(p) for p in prompts], device=dev)
    L0, Lmax = min(map(len, prompts)), max(map(len, prompts))
    total = Lmax + max_new
    assert total <= model.cfg.max_seq_len, "prompt + max_new exceeds max_seq_len"
    buf = torch.zeros(B, total, dtype=torch.long, device=dev)
    lpbuf = torch.zeros(B, total, device=dev)
    for i, p in enumerate(prompts):
        buf[i, : len(p)] = torch.tensor(p)
    cache = model.init_cache(B)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    with autocast(dev):
        logits = readout(model, model.hidden(buf[:, :L0], cache)[:, -1])
        for t in range(L0, total):
            logp = torch.log_softmax(logits.float() / temperature, -1)
            nxt = torch.multinomial(logp.exp(), 1, generator=gen)[:, 0]
            lpbuf[:, t] = logp.gather(1, nxt[:, None])[:, 0]
            gen_t = t >= lp
            buf[:, t] = torch.where(gen_t, nxt, buf[:, t])
            done |= gen_t & ((nxt == eot) | (t + 1 >= lp + max_new))
            # Poll every 16 steps only: each poll is a host-device sync.
            if t + 1 == total or (t % 16 == 0 and bool(done.all())):
                break
            logits = readout(model, model.hidden(buf[:, t : t + 1], cache)[:, -1])
    out, lps = [], []
    for i, (row, lrow) in enumerate(zip(buf.tolist(), lpbuf.tolist())):
        a = len(prompts[i])
        c = row[a : a + max_new]
        n = c.index(eot) + 1 if eot in c else len(c)
        out.append(c[:n])
        lps.append(lrow[a : a + n])
    return (out, lps) if return_logp else out


def completion_stats(model: GPT, prompts: list[list[int]], completions: list[list[int]],
                     entropy: bool = False) -> tuple[Tensor, Tensor, Tensor | None]:
    """Per-token log-probabilities (N, T), mask (N, T) and, optionally, per-token entropy in nats."""
    with autocast(model.tok_emb.weight.device):
        z, tgt, mask = completion_logits(model, prompts, completions)
    m = mask.float()
    lse = torch.logsumexp(z, -1)
    lp = (z.gather(-1, tgt[..., None])[..., 0] - lse) * m
    H = None
    if entropy:
        with torch.no_grad():
            H = (lse - (torch.softmax(z, -1) * z).sum(-1)) * m
    return lp, m, H


def decode(tok: Tokenizer, ids: list[int], eot: int) -> str:
    return tok.decode([i for i in ids if i != eot and i < tok.vocab_size]).decode("utf-8", errors="replace")


def score(headers: list[dict], texts: list[str], reward: str = "verse") -> tuple[Tensor, dict[str, Tensor]]:
    if reward == "naive":
        r = torch.tensor([naive_rhyme_reward(t) for t in texts])
        return r, {"naive_rhyme": r}
    rows = [verse_reward(h["lines"], t) for h, t in zip(headers, texts)]
    comp = {k: torch.tensor([r[k] for r in rows]) for k in rows[0]}
    return comp["total"], comp


def encode_headers(headers: list[dict], tok: Tokenizer) -> list[list[int]]:
    return [tok.encode(header_text(h, tok)) for h in headers]


def save_json_lines(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@torch.no_grad()
def stats_batched(model: GPT, prompts, comps, micro: int, entropy: bool = False) -> list[Tensor | None]:
    """completion_stats in micro-batches, without grad. (N, T) tensors padded to a common T."""
    out = [completion_stats(model, prompts[i : i + micro], comps[i : i + micro], entropy)
           for i in range(0, len(prompts), micro)]
    T = max(o[0].shape[1] for o in out)
    pad = lambda x: torch.nn.functional.pad(x, (0, T - x.shape[1]))
    return [torch.cat([pad(o[k]) for o in out]) if out[0][k] is not None else None for k in range(3)]


def grpo_update(policy: GPT, ref: GPT, opt, prompts, comps, R: Tensor, aggregation: str = "sequence",
                scale: str = "std", beta: float = 0.0, eps_high: float = 0.2, mu: int = 1, micro: int = 32,
                max_len: int | None = None, sampler_logp: list[list[float]] | None = None) -> dict:
    """One GRPO step on P groups of G rollouts. R: (P, G) rewards, in the order of prompts and comps."""
    dev = policy.tok_emb.weight.device
    A = group_advantages(R, scale=scale).flatten().to(dev)
    logp_old, mask, ent = stats_batched(policy, prompts, comps, micro, entropy=True)
    # Compute the reference also with beta = 0: the KL to it is a logged statistic.
    logp_ref = stats_batched(ref, prompts, comps, micro)[0]
    N, T = mask.shape
    st = {"ratio_err": 0.0, "loss": 0.0}
    if sampler_logp is not None:
        s = torch.zeros_like(logp_old)
        for i, l in enumerate(sampler_logp):
            s[i, : len(l)] = torch.tensor(l)
        # The KV-cache sampler and the full forward are two code paths. In bf16 they differ a little.
        st["sampler_err"] = ((s - logp_old).abs() * mask).max().item()
    for inner in range(mu):
        for i in range(0, N, micro):
            sl = slice(i, i + micro)
            lp, _, _ = completion_stats(policy, prompts[sl], comps[sl])
            lp, m = torch.nn.functional.pad(lp, (0, T - lp.shape[1])), mask[sl]
            if inner == 0:
                # Before the first update the ratio must be exactly 1. Anything else is a code-path mismatch.
                st["ratio_err"] = max(st["ratio_err"], ((lp.detach() - logp_old[sl]).abs() * m).max().item())
            loss = policy_loss(lp, logp_old[sl], A[sl], m, 0.2, eps_high, aggregation, logp_ref[sl], beta, max_len)
            # Each aggregation is a sum over items, over a batch-wide denominator. Re-weight the micro-batch.
            w = (m.sum() / mask.sum()).item() if aggregation == "token" else m.shape[0] / N
            (loss * w).backward()
            st["loss"] += loss.item() * w / mu
        st["grad_norm"] = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
    k1 = (kl_estimators(logp_old, logp_ref)["k1"] * mask).sum(1)
    return {**st, "kl": k1.mean().item(), "kl_se": (k1.std() / math.sqrt(N)).item(),
            "entropy": ((ent * mask).sum() / mask.sum()).item(), "len": mask.sum(1).mean().item()}
