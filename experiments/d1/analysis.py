"""D1 Task 4. Entropy per position, control accuracy of <|lines|>, memorisation."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lm.tokenizer.bpe import Tokenizer
from lm.verse.corpus import load_verse
from lm.model import GPT
from experiments.d1.run import load_base
from lm.model.checkpoint import load_model
from lm.data.sft import count_lines, header


def load_sft(path: str, tok_path: str) -> tuple[GPT, Tokenizer]:
    model, extra = load_model(path)
    return model.eval(), Tokenizer(Tokenizer.load(tok_path).merges, extra["special"])


@torch.no_grad()
def sample(model: GPT, prompts: list[list[int]], max_new: int, eot: int, temperature: float = 1.0,
           seed: int = 0, batch: int = 64) -> list[list[int]]:
    """KV-cached sampling. Prompts of the same length share a batch, so no padding mask is needed."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    out: list[list[int] | None] = [None] * len(prompts)
    by_len: dict[int, list[int]] = {}
    for i, p in enumerate(prompts):
        by_len.setdefault(len(p), []).append(i)
    for L, members in by_len.items():
        n_new = min(max_new, model.cfg.max_seq_len - L)
        for s in range(0, len(members), batch):
            ids = members[s : s + batch]
            x = torch.tensor([prompts[i] for i in ids], device="cuda")
            cache = model.init_cache(len(ids))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x, cache=cache)[:, -1].float()
            gen = torch.empty(len(ids), n_new, dtype=torch.long, device="cuda")
            done = torch.zeros(len(ids), dtype=torch.bool, device="cuda")
            for t in range(n_new):
                if temperature == 0:
                    nxt = logits.argmax(-1)
                else:
                    # Gumbel-max: argmax(z / tau + G) is a sample of softmax(z / tau).
                    u = torch.rand(logits.shape, generator=g, device="cuda").clamp_(min=1e-20)
                    nxt = (logits / temperature - (-u.log()).log()).argmax(-1)
                gen[:, t] = nxt
                done |= nxt == eot
                if bool(done.all()) or t == n_new - 1:
                    break
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(nxt[:, None], cache=cache)[:, -1].float()
            for j, i in enumerate(ids):
                row = gen[j, : t + 1].tolist()
                out[i] = row[: row.index(eot)] if eot in row else row
    return out


def longest_common_substring(a: list[int], b: list[int]) -> int:
    """Row-by-row DP, vectorised over b. O(len(a) len(b)) work, O(len(b)) memory."""
    if not a or not b:
        return 0
    bb = np.asarray(b)
    prev = np.zeros(len(b) + 1, dtype=np.int32)
    best = 0
    for x in a:
        cur = np.zeros_like(prev)
        cur[1:] = np.where(bb == x, prev[:-1] + 1, 0)
        best = max(best, int(cur.max()))
        prev = cur
    return best


def text(tok: Tokenizer, ids: list[int]) -> str:
    return tok.decode(ids).decode("utf-8", errors="replace")


def entropy_plot(log: Path, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    recs = [json.loads(l) for l in log.read_text().splitlines()]
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in recs:
        b = np.asarray(r["entropy_bins"])
        x = (np.arange(len(b)) + 0.5) / len(b)
        ax.plot(x, b, label="base" if r["step"] == 0 else f"step {r['step']}", c="k" if r["step"] == 0 else None)
    ax.set_xlabel("relative position in the response")
    ax.set_ylabel("mean predictive entropy (nats)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main(args):
    out = Path(args.out)
    entropy_plot(Path(args.log), out / "entropy_per_position.png")
    sft, tok = load_sft(args.sft, args.tok)
    base, btok = load_base(args.base).eval(), Tokenizer.load(args.tok)
    eot = tok.special["<|endoftext|>"]
    verse = load_verse(args.verse)
    res: dict = {"control": {}, "memorisation": {}}
    for n in args.lines:
        metas = [verse["val"][i % len(verse["val"])][0] for i in range(args.n_control)]
        for name, model, t, plain in (("sft", sft, tok, False), ("base_plain_header", base, btok, True)):
            gens = sample(model, [t.encode(header(m, n, plain)) for m in metas], args.max_new, eot, 1.0, seed=n)
            got = np.array([count_lines(text(t, g)) for g in gens])
            res["control"].setdefault(name, {})[n] = {"acc_pm1": float((np.abs(got - n) <= 1).mean()),
                                                      "mean_lines": float(got.mean())}
        print(n, {k: v[n] for k, v in res["control"].items()}, flush=True)
    for split in ("train", "val"):
        items = verse[split][: args.n_mem]
        prompts = [tok.encode(header(m, count_lines(s))) for m, s in items]
        gens = sample(sft, prompts, args.max_new, eot, temperature=0)
        lcs = [longest_common_substring(g, tok.encode(s)) for g, (_, s) in zip(gens, items)]
        res["memorisation"][split] = {"lcs_mean": float(np.mean(lcs)), "lcs_median": float(np.median(lcs)),
                                      "lcs_max": int(np.max(lcs))}
    recs = [json.loads(l) for l in Path(args.log).read_text().splitlines()]
    res["final_train_val_gap"] = recs[-1]["val_loss"] - recs[-1]["train_loss"]
    print(json.dumps(res, indent=1))
    (out / "analysis.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default="runs/d1/sft_lam0.0.pt")
    ap.add_argument("--log", default="runs/d1/sft_lam0.0.jsonl")
    ap.add_argument("--base", default="runs/c1_tier1/model.pt")
    ap.add_argument("--tok", default="data/fineweb_tok.json")
    ap.add_argument("--verse", default="data/verse.jsonl")
    ap.add_argument("--out", default="runs/d1")
    ap.add_argument("--lines", type=int, nargs="+", default=[4, 8, 14, 20])
    ap.add_argument("--n_control", type=int, default=50)
    ap.add_argument("--n_mem", type=int, default=50)
    ap.add_argument("--max_new", type=int, default=480)
    main(ap.parse_args())
