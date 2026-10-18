"""C2 Tasks 2 and 4 on the FineWeb-Edu sample: near-duplicate clusters, then contamination under two splits.

python -m experiments.c2.fineweb [--docs 20000]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from lm.data.corpus import MinHasher, assign_split, near_duplicate_clusters, shingles

EOT = "<|endoftext|>"


def contamination(cid: np.ndarray, split: np.ndarray) -> float:
    """Fraction of validation documents whose cluster also has a training document."""
    train_clusters = set(cid[split == "train"].tolist())
    val = cid[split == "val"]
    return float(np.mean([c in train_clusters for c in val.tolist()])) if len(val) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt", default="data/fineweb.txt")
    ap.add_argument("--docs", type=int, default=20_000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--b", type=int, default=14)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    docs = [d for d in Path(args.txt).read_text(encoding="utf-8").split(EOT) if d.strip()][: args.docs]
    t0 = time.perf_counter()
    mh = MinHasher(num_perm=args.b * args.r, seed=args.seed)
    sh = [shingles(d, args.k) for d in docs]
    sig = np.stack([mh.signature(s) for s in sh])
    t_sig = time.perf_counter() - t0
    cid = near_duplicate_clusters(sig, args.b, args.r)
    n_shingles = sum(map(len, sh))
    sizes = np.bincount(cid)
    in_multi = int((sizes[cid] > 1).sum())
    print(f"{len(docs)} documents, {n_shingles / 1e6:.1f} M shingles, signatures in {t_sig:.0f}s "
          f"({n_shingles / t_sig / 1e6:.2f} M shingles/s); clusters in {time.perf_counter() - t0 - t_sig:.1f}s")
    print(f"{in_multi} documents ({in_multi / len(docs):.2%}) are in {int((sizes > 1).sum())} multi-document clusters; "
          f"largest cluster {sizes.max()}")

    # Document-level split: a hash of the document index. Cluster-level: a hash of the cluster id.
    doc_split = assign_split(np.arange(len(docs)), args.val_frac, args.seed)
    clu_split = assign_split(cid, args.val_frac, args.seed)
    rec = {"docs": len(docs), "dup_docs": in_multi, "doc_split_contamination": contamination(cid, doc_split),
           "cluster_split_contamination": contamination(cid, clu_split), "b": args.b, "r": args.r, "k": args.k}
    print(f"validation documents with a near-duplicate in train: document-level split "
          f"{rec['doc_split_contamination']:.2%}, cluster-level split {rec['cluster_split_contamination']:.2%}")
    Path("runs/c2_fineweb.json").write_text(json.dumps(rec, indent=1))
    big = np.argsort(-sizes)[:3]
    for c in big:
        i = np.flatnonzero(cid == c)[:2]
        print(f"\n=== cluster of {sizes[c]}:\n  " + "\n  ".join(docs[j][:150].replace("\n", " / ") for j in i))


if __name__ == "__main__":
    main()
