"""A2 Tasks 5 and 6. Run from the repo root after experiments/a2/train_tok.py."""
import time
from collections import Counter
from pathlib import Path

from lm.tokenizer.bpe import Tokenizer, pretokenize
from lm.tokenizer.unigram import bpb, em_step, log_total, surface_logphi, viterbi


def compare(tok: Tokenizer, files: dict[str, str]) -> None:
    import tiktoken

    encs = {"ours-16k": lambda t: tok.encode(t)}
    for name in ("gpt2", "cl100k_base", "o200k_base"):
        e = tiktoken.get_encoding(name)
        encs[name] = lambda t, e=e: e.encode(t, disallowed_special=())
    print(f"{'tokeniser':>12} " + " ".join(f"{k + ' B/tok':>12} {k + ' tok/w':>12}" for k in files))
    for name, enc in encs.items():
        cells = []
        for text in files.values():
            t0 = time.perf_counter()
            n = len(enc(text))
            dt = time.perf_counter() - t0
            cells.append(f"{len(text.encode()) / n:>12.3f} {n / len(text.split()):>12.3f}")
        print(f"{name:>12} " + " ".join(cells) + f"   ({dt:.2f}s on the last file)")


def fibre_gap(tok: Tokenizer, train: str, test: str) -> None:
    counts = Counter(tok.encode(train))
    logphi = surface_logphi(tok.vocab[: 256 + len(tok.merges)], counts)
    max_len = max(len(s) for s in logphi)
    chunks = Counter(c.encode("utf-8") for c in pretokenize(test))
    n_bytes = sum(len(c) * f for c, f in chunks.items())

    canon = total = 0.0
    differ = 0
    for c, f in chunks.items():
        ids = tok.encode(c.decode("utf-8"))
        canon += f * sum(logphi[tok.vocab[i]] for i in ids)
        total += f * log_total(c, logphi, max_len)
        differ += f * (viterbi(c, logphi, max_len) != [tok.vocab[i] for i in ids])
    print(f"canonical {bpb(canon, n_bytes):.4f} bpb | total {bpb(total, n_bytes):.4f} bpb | "
          f"gap {bpb(canon, n_bytes) - bpb(total, n_bytes):.4f} | Viterbi != BPE on {differ / chunks.total():.1%} of chunks")

    sub = Counter(dict(Counter(c.encode("utf-8") for c in pretokenize(train)).most_common(5000)))
    _, ll0 = em_step(sub, logphi, max_len)
    phi1, _ = em_step(sub, logphi, max_len)
    _, ll1 = em_step(sub, phi1, max_len)
    print(f"EM on top-5000 train chunks: loglik {ll0:.1f} -> {ll1:.1f} (Proposition 4: must not decrease)")


def main():
    tok = Tokenizer.load("data/tok16k.json")
    en = Path("data/corpus.txt").read_text(encoding="utf-8")
    fr = Path("data/fr.txt").read_text(encoding="utf-8")
    compare(tok, {"EN": en, "FR": fr})
    n = len(en)
    fibre_gap(tok, en[: int(0.9 * n)], en[int(0.9 * n) :])


if __name__ == "__main__":
    main()
