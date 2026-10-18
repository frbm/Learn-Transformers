# tests/test_c2_data.py
import numpy as np, pytest, unicodedata
from lm.data.corpus import (normalize_text, dedup_key, shingles, MinHasher, jaccard, lsh_candidate_prob,
                  near_duplicate_clusters, assign_split, write_tokens, TokenFile, epoch_starts)

rng = np.random.default_rng(0)
WORDS = [f"w{i}" for i in range(5000)]

def doc(n=200):
    return " ".join(rng.choice(WORDS, n))

def perturb(s, frac):
    w = s.split()
    for i in rng.choice(len(w), int(frac * len(w)), replace=False): w[i] = str(rng.choice(WORDS))
    return " ".join(w)

def test_normalisations():
    assert dedup_key("Café   au LAIT,\n\n  encore!") == dedup_key(unicodedata.normalize("NFD", "café au lait encore"))
    assert normalize_text("x\r\ny  \n\n\n\nz") == "x\ny\n\nz"

def test_minhash_estimates_jaccard():
    mh = MinHasher(num_perm=256, seed=1)
    errs = []
    for frac in np.linspace(0.05, 0.9, 50):
        a = doc(); b = perturb(a, frac)
        sa, sb = shingles(a, 3), shingles(b, 3)
        errs.append(np.mean(mh.signature(sa) == mh.signature(sb)) - jaccard(sa, sb))
    errs = np.abs(errs)
    assert errs.mean() < 0.04 and errs.max() < 0.15

def test_lsh_probability():
    assert lsh_candidate_prob(0.8, b=20, r=5) == pytest.approx(1 - (1 - 0.8 ** 5) ** 20)
    assert lsh_candidate_prob(0.3, b=20, r=5) < 0.05 and lsh_candidate_prob(0.8, b=20, r=5) > 0.999

def test_clusters_recover_planted_near_duplicates():
    base = [doc() for _ in range(300)]
    docs = base + [perturb(base[i], 0.01) for i in range(0, 60, 2)]       # 30 planted near-duplicates, J ≈ 0.9
    mh = MinHasher(num_perm=112, seed=0)
    cid = near_duplicate_clusters(np.stack([mh.signature(shingles(d, 5)) for d in docs]), b=14, r=8)
    assert sum(cid[300 + j] == cid[2 * j] for j in range(30)) >= 28
    assert len(set(cid[:300])) == 300                                     # no false merges

def test_split_is_constant_on_clusters():
    cid = np.repeat(np.arange(500), 3)
    split = assign_split(cid, val_frac=0.1, seed=0)
    assert all(len(set(split[cid == c])) == 1 for c in range(500))
    assert 0.05 < np.mean(split == "val") < 0.15

def test_token_file_roundtrip(tmp_path):
    docs = [rng.integers(0, 16447, n).astype(np.uint16) for n in (5, 1, 300, 42)]
    write_tokens(docs, str(tmp_path / "toy"), eot=16447)
    tf = TokenFile(str(tmp_path / "toy"))
    assert len(tf) == 4 and tf.tokens.dtype == np.uint16
    for i, d in enumerate(docs): np.testing.assert_array_equal(tf.doc(i), d)
    with pytest.raises(ValueError):
        write_tokens([np.array([70_000])], str(tmp_path / "bad"), eot=0)

def test_epoch_tiles_the_stream():
    n, T = 10_000, 64
    s = epoch_starts(n, T, epoch=3, seed=0)
    cover = np.zeros(n, int)
    for x in s: cover[x:x + T] += 1
    assert cover.max() == 1 and cover.sum() >= n - 2 * T and np.all(s + T + 1 <= n)
    assert not np.array_equal(s, epoch_starts(n, T, epoch=4, seed=0))
