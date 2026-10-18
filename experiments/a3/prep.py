"""A3 Task 0. Download 100 MB of FineWeb-Edu, train the tokeniser, write uint16 ids and a meta file."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from lm.tokenizer.bpe import Tokenizer, train_bpe

EOT = "<|endoftext|>"


def download(path: Path, n_bytes: int) -> None:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in ds:
            f.write(row["text"] + EOT)
            n += len(row["text"].encode("utf-8"))
            if n > n_bytes:
                break


def encode_docs(tok: Tokenizer, text: str) -> np.ndarray:
    ids: list[int] = []
    eot = tok.special[EOT]
    for doc in text.split(EOT):
        if doc:
            ids.extend(tok.encode(doc))
            ids.append(eot)
    return np.asarray(ids, dtype=np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt", default="data/fineweb.txt")
    ap.add_argument("--out", default="data/fineweb")
    ap.add_argument("--bytes", type=int, default=100_000_000)
    ap.add_argument("--vocab", type=int, default=16_384, help="total, special token included")
    ap.add_argument("--tok_bytes", type=int, default=20_000_000)
    args = ap.parse_args()

    txt = Path(args.txt)
    if not txt.exists():
        download(txt, args.bytes)
    text = txt.read_text(encoding="utf-8")

    t0 = time.perf_counter()
    # Remove the specials before training. Merges must never see them.
    merges = train_bpe(text[: args.tok_bytes].replace(EOT, "\n"), args.vocab - 1)
    tok = Tokenizer(merges, [EOT])
    tok.save(f"{args.out}_tok.json")
    print(f"tokeniser: {tok.vocab_size} ids in {time.perf_counter() - t0:.0f}s")

    t0 = time.perf_counter()
    ids = encode_docs(tok, text)
    assert int(ids.max()) < 2**16 and tok.vocab_size <= 2**16
    ids.tofile(f"{args.out}.bin")
    split = int(0.99 * len(ids))
    val_bytes = len(tok.decode(ids[split:].tolist()))
    meta = {"n_tokens": len(ids), "split": split, "val_tokens": len(ids) - split, "val_bytes": val_bytes,
            "vocab_size": tok.vocab_size, "eot": tok.special[EOT]}
    Path(f"{args.out}_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"encoded {len(ids):,} tokens in {time.perf_counter() - t0:.0f}s; {meta}")


if __name__ == "__main__":
    main()
