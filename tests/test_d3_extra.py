import math

import pytest
import torch

from lm.rl.policy import completion_stats, sample
from lm.model import GPT, Config
from lm.rl.prefs import completion_logprobs, dpo_loss, ipo_loss, policy_loss


def tiny(seed=0, V=97):
    torch.manual_seed(seed)
    return GPT(Config(vocab_size=V, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64)).double().eval()


def test_completion_logprobs_match_one_sequence_at_a_time():
    model = tiny()
    prompts, comps = [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10]], [[11, 12], [13, 14, 15, 16], [17]]
    with torch.no_grad():
        got = completion_logprobs(model, prompts, comps)
        lp, mask = completion_logprobs(model, prompts, comps, per_token=True)
        for i, (p, c) in enumerate(zip(prompts, comps)):
            x = torch.tensor(p + c)[None]
            ls = torch.log_softmax(model(x)[0], -1)
            want = ls[torch.arange(len(p) - 1, len(p) + len(c) - 1), torch.tensor(c)]
            torch.testing.assert_close(got[i], want.sum())
            torch.testing.assert_close(lp[i, : len(c)], want)
    assert mask.sum().item() == 7


def test_sampler_handles_ragged_prompts_like_greedy_generate():
    """Two code paths: the ragged teacher-forced cached sampler at T -> 0, and GPT.generate one prompt at a time."""
    model = tiny(1)
    prompts = [[1, 2, 3, 4, 5], [6, 7], [8, 9, 10]]
    out = sample(model, prompts, max_new=6, eot=-1, temperature=1e-6)
    for p, c in zip(prompts, out):
        want = model.generate(torch.tensor(p)[None], 6, use_cache=False)[0, len(p):].tolist()
        assert c == want


def test_sampler_stops_at_eot_and_logprobs_agree():
    model = tiny(2, V=7)
    g = torch.Generator().manual_seed(0)
    out = sample(model, [[1, 2]] * 64, max_new=20, eot=0, gen=g)
    assert all(0 not in c[:-1] for c in out) and any(c[-1] == 0 and len(c) < 20 for c in out)
    lp, m, H = completion_stats(model, [[1, 2]] * 64, out, entropy=True)
    assert (H[m.bool()] <= math.log(7) + 1e-9).all() and (lp <= 0).all()


def test_sampler_frequencies_match_the_model():
    model = tiny(3, V=5)
    g = torch.Generator().manual_seed(0)
    out = sample(model, [[1, 2, 3]] * 20000, max_new=1, eot=-1, gen=g)
    freq = torch.bincount(torch.tensor([c[0] for c in out]), minlength=5).double() / 20000
    with torch.no_grad():
        p = torch.softmax(model(torch.tensor([[1, 2, 3]]))[0, -1], -1)
    assert (freq - p).abs().max() < 4 * math.sqrt(0.25 / 20000)


def test_ipo_is_minimal_at_the_target_margin_and_dpo_has_no_minimiser():
    beta = 0.5
    h = torch.tensor([1 / (2 * beta)], dtype=torch.float64, requires_grad=True)
    z = torch.zeros(1, dtype=torch.float64)
    ipo_loss(h, z, z, z, beta).backward()
    assert h.grad.abs().item() < 1e-12
    margins = torch.tensor([1.0, 10.0, 100.0], dtype=torch.float64)
    losses, _ = dpo_loss(margins, torch.zeros(3, dtype=torch.float64), z.expand(3), z.expand(3), 1.0, reduction="none")
    assert (losses.diff() < 0).all() and (losses > 0).all()          # Proposition 3: decreasing, never zero


def test_policy_loss_at_ratio_one_is_the_policy_gradient():
    torch.manual_seed(0)
    logp = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.float64)
    adv = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64)
    policy_loss(logp, logp.detach(), adv, mask, aggregation="token").backward()
    torch.testing.assert_close(logp.grad, -adv[:, None] * mask / mask.sum())


def test_rewards_basics():
    pytest.importorskip("pronouncing")
    from lm.verse.rewards import naive_rhyme_reward, rhyme_part, syllable_count, verse_reward
    assert syllable_count("little") == 2 and syllable_count("Qwzzx") is None
    assert rhyme_part("Hat!") == rhyme_part("cat")
    assert naive_rhyme_reward("The cat\nThe cat\nThe cat") == 1.0               # the hack the naive reward allows
    assert verse_reward(4, "")["total"] == 0.0
    r = verse_reward(8, "one line only\nand another")
    assert all(0 <= v <= 1 for v in r.values())


def test_grpo_update_learns_a_toy_reward():
    """End to end: sampler -> group advantages -> clipped loss. Reward: fraction of token 3 in the completion."""
    from lm.rl.policy import frozen_copy, grpo_update
    model = tiny(4, V=8)
    ref = frozen_copy(model)
    opt = torch.optim.Adam(model.parameters(), lr=3e-2)
    g = torch.Generator().manual_seed(0)
    P, G, rewards = 4, 8, []
    for step in range(30):
        prompts = [[1, 2]] * (P * G)
        comps, lps = sample(model, prompts, max_new=6, eot=-1, gen=g, return_logp=True)
        R = torch.tensor([sum(t == 3 for t in c) / len(c) for c in comps], dtype=torch.float64).view(P, G)
        st = grpo_update(model, ref, opt, prompts, comps, R, aggregation="token", beta=0.0, micro=16,
                         sampler_logp=lps)
        assert st["ratio_err"] == 0.0 and st["sampler_err"] < 1e-4
        rewards.append(R.mean().item())
    assert sum(rewards[-5:]) / 5 > sum(rewards[:5]) / 5 + 0.3
