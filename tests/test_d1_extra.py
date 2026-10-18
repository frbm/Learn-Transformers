import math

import pytest
import torch

from lm.tokenizer.bpe import Tokenizer
from lm.model import GPT, Config
from lm.data.sft import (SPECIALS, Example, add_special_tokens, loss_denominator, pack, response_rel_pos, sft_loss,
                 sft_loss_sums, use_flex)
from test_d1_sft import rand_examples

CFG = dict(vocab_size=500, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=128)


@pytest.mark.parametrize("weighting", ["token", "sequence"])
def test_accumulated_sums_equal_full_batch(weighting):
    torch.manual_seed(0)
    b = pack(rand_examples(40), T=128, pad_id=0)
    logits = torch.randn(*b["labels"].shape, 11, dtype=torch.float64)
    lab = torch.where(b["labels"] == -100, -100, b["labels"] % 11)
    full = sft_loss(logits, lab, b["doc_ids"], weighting)
    den = loss_denominator(lab, b["doc_ids"], weighting)
    num = sum(sft_loss_sums(logits[i : i + 3], lab[i : i + 3], b["doc_ids"][i : i + 3], weighting)[0]
              for i in range(0, len(lab), 3))
    torch.testing.assert_close(num / den, full)


def test_add_special_tokens_kl_bound():                            # Proposition 3
    torch.manual_seed(0)
    tok = Tokenizer([], ["<|endoftext|>"])
    model = GPT(Config(**{**CFG, "vocab_size": tok.vocab_size})).double().eval()
    x = torch.randint(0, 256, (8, 32))
    with torch.no_grad():
        p = model(x).log_softmax(-1)
        add_special_tokens(model, tok, SPECIALS)
        q = model(x).log_softmax(-1)
    V, k = p.shape[-1], len(SPECIALS)
    assert q.shape[-1] == V + k and tok.special["<|body|>"] == V + k - 1
    assert q[..., V:].exp().max() <= 1 / V
    kl = (p.exp() * (p - q[..., :V])).sum(-1)
    assert kl.max() <= math.log(1 + k / V) + 1e-12
    assert tok.encode("<|lines|> 4<|body|>")[0] == tok.special["<|lines|>"]


def test_response_rel_pos():
    lab = torch.tensor([[-100, 5, 5, 5, 5, -100, 7, 7]])
    doc = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1]])
    want = torch.tensor([[-1, 0, 0.25, 0.5, 0.75, -1, 0, 0.5]], dtype=torch.float64)
    torch.testing.assert_close(response_rel_pos(lab, doc), want)


def test_pack_drops_long_examples():
    exs = [Example([1, 2], [3] * 10), Example([1], [4] * 200)]
    b = pack(exs, T=16, pad_id=0)
    assert int(b["n_dropped"]) == 1 and set(b["doc_ids"].unique().tolist()) == {-1, 0}


@pytest.mark.gpu
@pytest.mark.skipif(not use_flex("cuda") if torch.cuda.is_available() else True, reason="needs CUDA and triton")
def test_flex_matches_dense_mask():
    torch.manual_seed(0)
    model = GPT(Config(**CFG)).cuda().eval()
    b = {k: v.cuda() for k, v in pack(rand_examples(12, seed=2), T=128, pad_id=0).items()}
    with torch.no_grad():
        dense = model(b["input_ids"], doc_ids=b["doc_ids"], position_ids=b["position_ids"], flex=False)
        flex = model(b["input_ids"], doc_ids=b["doc_ids"], position_ids=b["position_ids"], flex=True)
    keep = b["doc_ids"] >= 0
    torch.testing.assert_close(flex[keep], dense[keep], atol=1e-4, rtol=1e-4)
