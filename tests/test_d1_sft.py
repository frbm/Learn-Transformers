# tests/test_d1_sft.py
import math, pytest, torch
from lm.model import Config, GPT
from lm.data.sft import Example, pack, block_causal_mask, sft_loss, entropy_from_logits

def rand_examples(n, seed=0, V=500):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        lp = int(torch.randint(2, 8, (1,), generator=g)); lr = int(torch.randint(3, 60, (1,), generator=g))
        out.append(Example(prompt=torch.randint(0, V, (lp,), generator=g).tolist(),
                           response=torch.randint(0, V, (lr,), generator=g).tolist()))
    return out

def test_pack_is_lossless_and_masked():
    exs, T = rand_examples(200), 128
    b = pack(exs, T=T, pad_id=0)
    ids, lab, doc, pos = b["input_ids"], b["labels"], b["doc_ids"], b["position_ids"]
    assert ids.shape[1] == T and ids.shape == lab.shape == doc.shape == pos.shape
    seen = set()
    for r in range(ids.shape[0]):
        for d in doc[r].unique().tolist():
            if d < 0: continue
            sel, ex = doc[r] == d, exs[d]
            full = ex.prompt + ex.response
            assert ids[r][sel].tolist() == full[:-1]
            assert lab[r][sel].tolist() == [-100] * (len(ex.prompt) - 1) + ex.response
            assert pos[r][sel].tolist() == list(range(len(full) - 1))
            seen.add(d)
        assert (lab[r][doc[r] < 0] == -100).all()
    assert seen == set(range(len(exs)))
    assert (doc >= 0).float().mean() > 0.9                     # first-fit decreasing utilisation

def test_block_causal_mask():
    m = block_causal_mask(torch.tensor([[0, 0, 1, 1, 1, -1]]))[0]
    want = torch.tensor([[1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                         [0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0], [0, 0, 0, 0, 0, 1]], dtype=torch.bool)
    assert torch.equal(m, want)

def test_packed_forward_equals_separate_forwards():               # Proposition 4
    torch.manual_seed(0)
    cfg = Config(vocab_size=500, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=256)
    model = GPT(cfg).double().eval()
    exs = rand_examples(3, seed=1)
    b = pack(exs, T=256, pad_id=0)
    assert b["input_ids"].shape[0] == 1
    with torch.no_grad():
        packed = model(b["input_ids"], doc_ids=b["doc_ids"], position_ids=b["position_ids"])[0]
        for d, ex in enumerate(exs):
            alone = model(torch.tensor(ex.prompt + ex.response)[:-1][None])[0]
            torch.testing.assert_close(packed[b["doc_ids"][0] == d], alone)

def test_sft_loss_weightings():                                    # Proposition 2
    torch.manual_seed(0)
    logits = torch.randn(1, 10, 7, dtype=torch.float64)
    labels = torch.tensor([[-100, 1, 2, 3, -100, -100, 4, -100, -100, -100]])
    doc = torch.tensor([[0, 0, 0, 0, 1, 1, 1, -1, -1, -1]])
    nll = -torch.log_softmax(logits[0], -1)[torch.arange(10), labels[0].clamp(min=0)]
    torch.testing.assert_close(sft_loss(logits, labels, doc, weighting="token"), nll[[1, 2, 3, 6]].mean())
    torch.testing.assert_close(sft_loss(logits, labels, doc, weighting="sequence"), (nll[[1, 2, 3]].mean() + nll[6]) / 2)

def test_entropy_from_logits():
    V = 50
    assert entropy_from_logits(torch.zeros(V)).item() == pytest.approx(math.log(V))
    peaked = torch.full((V,), -1e4); peaked[3] = 0.0
    assert entropy_from_logits(peaked).item() == pytest.approx(0.0, abs=1e-6)
