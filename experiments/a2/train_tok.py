import argparse
import time
from pathlib import Path

from lm.tokenizer.bpe import Tokenizer, train_bpe

SPECIAL = ["<|endoftext|>"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus.txt")
    ap.add_argument("--vocab", type=int, default=16_384)
    ap.add_argument("--out", default="data/tok16k.json")
    args = ap.parse_args()
    text = Path(args.corpus).read_text(encoding="utf-8")
    t0 = time.perf_counter()
    merges = train_bpe(text, args.vocab - len(SPECIAL))
    tok = Tokenizer(merges, SPECIAL)
    print(f"{len(merges)} merges in {time.perf_counter() - t0:.1f}s; vocab {tok.vocab_size}")
    tok.save(args.out)
    t0 = time.perf_counter()
    ids = tok.encode(text)
    print(f"encode: {time.perf_counter() - t0:.2f}s, {len(text.encode()) / len(ids):.2f} bytes/token, "
          f"cache {tok._encode_chunk.cache_info()}")


if __name__ == "__main__":
    main()
