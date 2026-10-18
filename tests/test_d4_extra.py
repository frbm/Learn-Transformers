import pytest, torch
from lm.model import Config, GPT
from lm.infer.sampling import Sampler, eta_cutoff, temperature, top_k, make_sampler
from lm.infer.generate import BatchCache, forward_cached, generate, sequence_logprob, beam_search, _graph_step
from lm.infer.speculative import speculative_cached

f64 = dict(dtype=torch.float64)


def tiny(V=64, d=32, layers=2, seed=0, T=64):
    torch.manual_seed(seed)
    return GPT(Config(vocab_size=V, d_model=d, n_layers=layers, n_heads=4, n_kv_heads=2, max_seq_len=T)).double().eval()


def test_padded_cached_forward_equals_plain_forward():           # two code paths
    m = tiny()
    prompts = [[1, 2, 3, 4, 5], [7, 8], [9, 10, 11]]
    from lm.infer.generate import left_pad
    idx, valid = left_pad(prompts, 0, "cpu")
    cache = BatchCache(m.cfg, 3, 16, "cpu", torch.float64)
    out = forward_cached(m, idx, cache, valid)
    with torch.no_grad():
        for i, p in enumerate(prompts):
            torch.testing.assert_close(out[i, -len(p):], m(torch.tensor([p]))[0])


def test_rollback_then_refeed():
    m = tiny()
    cache = BatchCache(m.cfg, 1, 32, "cpu", torch.float64)
    forward_cached(m, torch.tensor([[1, 2, 3]]), cache)
    forward_cached(m, torch.tensor([[4, 5, 6]]), cache)
    cache.rollback(4)                                         # keep 1 2 3 4
    got = forward_cached(m, torch.tensor([[9, 9]]), cache)
    with torch.no_grad():
        want = m(torch.tensor([[1, 2, 3, 4, 9, 9]]))[0, -2:]
    torch.testing.assert_close(got[0], want)


def test_batched_greedy_equals_single_stream():
    m = tiny()
    prompts = [[3, 4, 5, 6, 7, 8], [1], [2, 9, 13]]
    out = generate(m, prompts, 10, Sampler.greedy(), eos_id=-1)
    for p, o in zip(prompts, out):
        assert o == m.generate(torch.tensor([p]), 10)[0, len(p):].tolist()


def test_logq_matches_rescoring():                                # F1 Proposition 1, two code paths
    m = tiny()
    proc = lambda x: top_k(temperature(x, 0.8), 20)
    g = torch.Generator().manual_seed(0)
    prompts = [[3, 4, 5], [1, 2]]
    seqs, logq = generate(m, prompts, 12, Sampler([proc]), eos_id=0, generator=g, return_logq=True)
    resp = [s + [0] if len(s) < 12 else s for s in seqs]
    torch.testing.assert_close(sequence_logprob(m, prompts, resp, processor=proc), torch.tensor(logq, **f64))


def test_per_row_stop():
    m = tiny()
    with torch.no_grad():
        first = m(torch.tensor([[5]]))[0, -1].argmax().item()
    out = generate(m, [[5], [6]], 8, Sampler.greedy(), eos_id=first)
    assert out[0] == [] and len(out[1]) <= 8


def test_beam_one_is_greedy():
    m = tiny()
    assert beam_search(m, [3, 4], 1, 8, eos_id=-1) == generate(m, [[3, 4]], 8, Sampler.greedy(), eos_id=-1)[0]


def test_beam_finds_at_least_greedy_logprob():
    m = tiny(seed=3)
    b = beam_search(m, [3, 4], 4, 6, eos_id=-1)
    gr = generate(m, [[3, 4]], 6, Sampler.greedy(), eos_id=-1)[0]
    lp = lambda r: sequence_logprob(m, [[3, 4]], [r]).item()
    assert lp(b) >= lp(gr) - 1e-9


def test_eta_threshold():
    l = torch.randn(4, 100, **f64) * 2
    out = eta_cutoff(l, 1e-2)
    p = l.softmax(-1)
    H = -(p * p.log()).sum(-1, keepdim=True)
    thr = torch.minimum(torch.full_like(H, 1e-2), 0.1 * torch.exp(-H))
    assert torch.equal(torch.isfinite(out), (p > thr) | (p == p.amax(-1, keepdim=True)))


def test_make_sampler_order():
    s = make_sampler("temp=2.0,top_p=0.72")
    l = torch.log(torch.tensor([0.4, 0.35, 0.25], **f64))
    assert torch.isfinite(s.process(l)).sum() == 3


def test_speculative_cached_is_exact():                           # Proposition 5 on real (tiny) GPTs
    V = 4
    target, draft = tiny(V=V, seed=1), tiny(V=V, seed=2)
    for p in (*target.parameters(),):
        p.data.mul_(40)                                          # sharpen the target so p != q clearly
    sampler = Sampler([])
    prompt = [1, 2, 3]
    with torch.no_grad():
        p1 = target(torch.tensor([prompt]))[0, -1].softmax(-1)
        p2 = torch.stack([target(torch.tensor([prompt + [a]]))[0, -1].softmax(-1) for a in range(V)])
    exact = p1[:, None] * p2
    g = torch.Generator().manual_seed(0)
    n, counts, calls = 4000, torch.zeros(V, V, **f64), 0
    for _ in range(n):
        out, st = speculative_cached(target, draft, prompt, 2, 3, sampler, generator=g)
        counts[out[0], out[1]] += 1; calls += st["target_calls"]
    e = n * exact
    m = e > 5
    assert ((counts[m] - e[m]) ** 2 / e[m]).sum().item() < 37.70          # 15 dof at most, p = 0.001
    assert calls < 2 * n


def test_speculative_cached_long_run_with_self_draft():
    m = tiny(seed=4)
    out, st = speculative_cached(m, m, [1, 2], 30, 4, Sampler([]), generator=torch.Generator().manual_seed(0))
    assert len(out) == 30 and st["accepted"] == st["proposed"]   # q = p: every draft is accepted


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_graph_step_equals_eager():
    m = tiny(V=128, d=64, T=64).float().cuda()
    prompt = torch.randint(0, 128, (2, 5), device="cuda")
    c1 = BatchCache(m.cfg, 2, 20, "cuda", torch.float32)
    tok = forward_cached(m, prompt, c1)[:, -1].argmax(-1, keepdim=True)
    eager = [tok]
    for _ in range(8):
        eager.append(forward_cached(m, eager[-1], c1)[:, -1].argmax(-1, keepdim=True))
    c2 = BatchCache(m.cfg, 2, 20, "cuda", torch.float32)
    step = _graph_step(m, c2, 2, 5)
    graphed = [forward_cached(m, prompt, c2)[:, -1].argmax(-1, keepdim=True)]
    for _ in range(8):
        graphed.append(step(graphed[-1]).clone())
    assert torch.equal(torch.cat(eager, 1), torch.cat(graphed, 1))


def test_line_constraint_gives_exactly_n_lines():
    from types import SimpleNamespace
    from lm.verse.common import LineConstraint, count_lines
    vocab = [b"a", b" ", b"\n", b"a\nb", b"\n\n", b"b c", b"<|endoftext|>"] + [b"z"] * 9
    tok = SimpleNamespace(vocab=vocab)
    m = tiny(V=16, seed=5)
    eos, B = 6, 64
    for n in (1, 3):
        con = LineConstraint(tok, 16, torch.full((B,), n), eos)
        out = generate(m, [[0]] * B, 60, Sampler([lambda x: temperature(x, 3.0)]), eos, constraint=con,
                       generator=torch.Generator().manual_seed(n))
        finished = [o for o in out if len(o) < 60]
        assert len(finished) > B // 2
        for o in finished:
            assert count_lines(b"".join(vocab[i] for i in o).decode()) == n


def test_rescore_per_token_sums_to_total():
    m = tiny()
    r = __import__("lm.infer.generate", fromlist=["rescore"]).rescore(m, [[1, 2], [3]], [[4, 5, 6], [7]], per_token=True)
    assert [len(t) for t in r["tok_logp"]] == [3, 1]
    torch.testing.assert_close(torch.stack([t.sum() for t in r["tok_logp"]]), r["logp"])
    with torch.no_grad():
        want = m(torch.tensor([[3, 7]]))[0, 0].log_softmax(-1)[7]
    torch.testing.assert_close(r["tok_logp"][1][0], want)
