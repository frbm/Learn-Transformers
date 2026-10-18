"""C2 Task 5: tokeniser throughput (naive, pretoken cache, cache in a process pool), token files, Batches/s.

python -m experiments.c2.pack                 # data/verse_{train,val}.bin/.idx and data/verse_meta.json
python -m experiments.c2.pack --bench_mb 10   # the throughput table on 10 MB of FineWeb text
"""
import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path


from lm.data.corpus import Batches, TokenFile, write_tokens
from lm.tokenizer.bpe import Tokenizer, _PRETOK
from lm.tokenizer.parallel import encode_docs

def encode_naive(tok: Tokenizer, text: str) -> list[int]:
    """No cache: merge every pretoken from scratch."""
    out = []
    for c in _PRETOK.findall(text):
        out.extend(tok._encode_chunk_uncached(c))
    return out


def bench(tok_path: str, docs: list[str], workers: int) -> None:
    n_bytes = sum(len(d.encode()) for d in docs)
    tok = Tokenizer.load(tok_path)
    rows = []
    t0 = time.perf_counter()
    for d in docs:
        encode_naive(tok, d)
    rows.append(("naive", time.perf_counter() - t0, ""))
    t0 = time.perf_counter()
    for d in docs:
        tok.encode(d)
    info = tok._encode_chunk.cache_info()
    rows.append(("pretoken cache", time.perf_counter() - t0, f"hit rate {info.hits / (info.hits + info.misses):.1%}"))
    t0 = time.perf_counter()
    for _ in encode_docs(tok_path, docs, workers):
        pass
    rows.append((f"cache + {workers} workers", time.perf_counter() - t0, "includes pool start-up"))
    for name, dt, note in rows:
        print(f"{name:>22}: {n_bytes / dt / 1e6:6.2f} MB/s  {note}")


def pack(tok_path: str, verse: str, out: str, workers: int) -> None:
    rows = [json.loads(l) for l in open(verse, encoding="utf-8")]
    tok = Tokenizer.load(tok_path)
    eot = tok.special["<|endoftext|>"]
    meta = {"vocab_size": tok.vocab_size, "eot": eot, "tokenizer": tok_path}
    for split in ("train", "val"):
        texts = [r["text"] for r in rows if r["split"] == split]
        # imap keeps the order and streams: write_tokens consumes documents as they arrive.
        write_tokens(encode_docs(tok_path, texts, workers), f"{out}_{split}", eot)
        tf = TokenFile(f"{out}_{split}")
        meta[f"{split}_docs"], meta[f"{split}_tokens"] = len(tf), len(tf.tokens)
        meta[f"{split}_bytes"] = sum(len(t.encode()) for t in texts) + len(texts)  # + one byte per eot
    Path(f"{out}_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


def batches_per_s(prefix: str, B: int, T: int, device: str, n: int = 500) -> float:
    tokens = TokenFile(f"{prefix}_train").tokens
    it = Batches(tokens, B, T, device=device)
    next(it)
    t0 = time.perf_counter()
    for _ in range(n):
        x, _ = next(it)
    if device == "cuda":
        import torch

        torch.cuda.synchronize()
    return n / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", default="data/fineweb_tok.json")
    ap.add_argument("--verse", default="data/verse.jsonl")
    ap.add_argument("--out", default="data/verse")
    ap.add_argument("--workers", type=int, default=min(8, mp.cpu_count()))
    ap.add_argument("--bench_mb", type=float, default=0)
    args = ap.parse_args()
    if args.bench_mb:
        text = Path("data/fineweb.txt").read_text(encoding="utf-8")[: int(args.bench_mb * 1e6)]
        bench(args.tok, [d for d in text.split("<|endoftext|>") if d], args.workers)
        return
    pack(args.tok, args.verse, args.out, args.workers)
    for B, T in ((16, 1024), (32, 512)):
        r = batches_per_s(args.out, B, T, "cuda")
        print(f"Batches B={B} T={T}: {r:.0f} batches/s = {r * B * T / 1e6:.1f} M tokens/s")


if __name__ == "__main__":
    main()
