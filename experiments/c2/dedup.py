"""C2 Tasks 2 and 4 on the verse corpus: exact dedup, MinHash clusters, inspection, cluster-level split.

python -m experiments.c2.dedup                  # data/c2/verse_clean.jsonl -> data/verse.jsonl (D1's input)
python -m experiments.c2.dedup --inspect 20     # print 20 random multi-document clusters
"""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from lm.data.corpus import MinHasher, assign_split, dedup_key, jaccard, lsh_candidate_prob, near_duplicate_clusters, shingles

DIR = Path("data/c2")


def exact_dedup(docs: list[dict]) -> list[dict]:
    """Keep the first document of each dedup key. blake2b, not hash(): Python's str hash is salted per process."""
    seen, out = set(), []
    for d in docs:
        h = hashlib.blake2b(dedup_key(d["text"]).encode(), digest_size=8).digest()
        if h not in seen:
            seen.add(h)
            out.append(d)
    return out


def minhash_clusters(texts: list[str], k: int, b: int, r: int, seed: int = 0) -> tuple[np.ndarray, list[np.ndarray]]:
    mh = MinHasher(num_perm=b * r, seed=seed)
    sh = [shingles(t, k) for t in texts]
    sig = np.stack([mh.signature(s) for s in sh])
    return near_duplicate_clusters(sig, b, r), sh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5, help="words per shingle")
    ap.add_argument("--b", type=int, default=13)
    ap.add_argument("--r", type=int, default=9)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inspect", type=int, default=0)
    ap.add_argument("--out", default="data/verse.jsonl")
    args = ap.parse_args()

    docs = [json.loads(l) for l in open(DIR / "verse_clean.jsonl", encoding="utf-8")]
    funnel = [("clean", len(docs))]
    docs = exact_dedup(docs)
    funnel.append(("exact dedup", len(docs)))
    cid, sh = minhash_clusters([d["text"] for d in docs], args.k, args.b, args.r, args.seed)
    members = defaultdict(list)
    for i, c in enumerate(cid):
        members[c].append(i)
    multi = [m for m in members.values() if len(m) > 1]
    print(f"(b, r) = ({args.b}, {args.r}): P(candidate) = {lsh_candidate_prob(0.85, args.b, args.r):.3f} at J = 0.85, "
          f"{lsh_candidate_prob(0.5, args.b, args.r):.3f} at J = 0.5, threshold ~ {(1 / args.b) ** (1 / args.r):.2f}")
    print(f"{len(multi)} multi-document clusters, {sum(map(len, multi))} documents in them")

    if args.inspect:
        for m in random.Random(args.seed).sample(multi, min(args.inspect, len(multi))):
            print(f"\n=== cluster of {len(m)}")
            for i in m[:3]:
                d = docs[i]
                print(f"  {d['source']:>16} {d['title'][:40]!r:44} {d['author'][:24]!r:28} "
                      f"J to first = {jaccard(sh[i], sh[m[0]]):.2f}")
            print("  " + docs[m[0]]["text"][:160].replace("\n", " / "))

    # Keep one member per cluster: an explicitly attributed copy first (an anthology without attribution lines
    # credits every poem to its compiler), then the longest. Then split by cluster id: no near-copy crosses it.
    attributed = lambda d: d["author"] != d.get("book_author") or "Various" not in d.get("book_author", "")
    keep = [max(m, key=lambda i: (attributed(docs[i]), len(docs[i]["text"]))) for m in members.values()]
    keep.sort()
    split = assign_split(cid[keep], args.val_frac, args.seed)
    funnel.append(("near dedup", len(keep)))
    out = [{**docs[i], "cluster": int(cid[i]), "split": str(s)} for i, s in zip(keep, split)]
    Path(args.out).write_text("".join(json.dumps(d) + "\n" for d in out), encoding="utf-8")
    words = lambda ds: sum(len(d["text"].split()) for d in ds)
    print("\nfunnel: " + " -> ".join(f"{n} {c}" for n, c in funnel))
    print(f"split: {sum(s == 'train' for s in split)} train, {sum(s == 'val' for s in split)} val; "
          f"{words(out) / 1e6:.2f} M words -> {args.out}")


if __name__ == "__main__":
    main()
