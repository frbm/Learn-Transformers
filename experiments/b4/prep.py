"""B4 setup. Download ~800 MB of FineWeb-Edu and encode it with the A2/A3 tokeniser on all cores."""
import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from lm.tokenizer.bpe import Tokenizer
from lm.tokenizer.parallel import encode_batches
from experiments.a3.prep import EOT, download

def doc_batches(path: Path, batch_bytes: int = 4_000_000):
    """Yield lists of documents. Read the file in blocks, never as one string."""
    buf, batch, size = "", [], 0
    with open(path, encoding="utf-8") as f:
        while block := f.read(1 << 22):
            *docs, buf = (buf + block).split(EOT)
            for d in docs:
                if d:
                    batch.append(d); size += len(d)
                if size >= batch_bytes:
                    yield batch
                    batch, size = [], 0
    if buf:
        batch.append(buf)
    if batch:
        yield batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt", default="data/fineweb_b4.txt")
    ap.add_argument("--out", default="data/fineweb_b4")
    ap.add_argument("--tok", default="data/fineweb_tok.json")
    ap.add_argument("--bytes", type=int, default=800_000_000)
    ap.add_argument("--workers", type=int, default=mp.cpu_count())
    args = ap.parse_args()

    txt = Path(args.txt)
    if not txt.exists():
        download(txt, args.bytes)
    tok = Tokenizer.load(args.tok)
    # imap keeps the document order. Then the split stays contiguous and reproducible.
    with open(f"{args.out}.bin", "wb") as f:
        n = 0
        for i, ids in enumerate(encode_batches(args.tok, doc_batches(txt), args.workers)):
            ids.tofile(f)
            n += len(ids)
            if i % 20 == 0:
                print(f"{n / 1e6:.0f}M tokens", flush=True)
    data = np.memmap(f"{args.out}.bin", dtype=np.uint16, mode="r")
    split = int(0.99 * len(data))
    eot = tok.special[EOT]
    val = data[split:]
    val_bytes = len(tok.decode([int(t) for t in val if t != eot]))
    meta = {"n_tokens": len(data), "split": split, "val_tokens": len(val), "val_bytes": val_bytes,
            "vocab_size": tok.vocab_size, "eot": eot, "tokenizer": args.tok}
    Path(f"{args.out}_meta.json").write_text(json.dumps(meta, indent=1))
    print(meta)


if __name__ == "__main__":
    main()
