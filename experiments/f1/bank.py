"""F1 Task 2: one sample bank for every system. One JSONL file per system; never regenerate.

python -m experiments.f1.bank --systems runs/f1/systems.json      # writes a template on first use
python -m experiments.f1.bank --smoke                             # random model, 3 systems, 4 headers x 4 samples

A system: {"name", "ckpt", "sampler", "best_of": n (optional), "steer": {"vec", "layer", "alpha"} (optional),
           "instruction": str (optional), "policy": true if held-out bpb applies (a trained model, not a decoder)}.
"""
import argparse
import contextlib
import json
import math
from pathlib import Path

import torch

from lm.verse.common import count_lines, load_model, load_reward, load_tokenizer, steer, verse_split, write_jsonl
from lm.eval.embedding import sample_set
from lm.infer.sampling import make_sampler

SFT, D3 = "runs/d1/sft.pt", "runs/d3"
TEMPLATE = [
    *[{"name": f"sft_T{t}", "ckpt": SFT, "sampler": f"temp={t}", "policy": t == 1.0} for t in (0.7, 1.0, 1.2)],
    {"name": "sft_top_p0.9", "ckpt": SFT, "sampler": "top_p=0.9"},
    {"name": "sft_min_p0.1", "ckpt": SFT, "sampler": "min_p=0.1"},
    *[{"name": f"dpo_beta{b}", "ckpt": f"{D3}/dpo_beta{b}.pt", "sampler": "temp=1.0", "policy": True} for b in (0.1, 0.5)],
    {"name": "grpo", "ckpt": f"{D3}/grpo.pt", "sampler": "temp=1.0", "policy": True},
    {"name": "dapo", "ckpt": f"{D3}/dapo.pt", "sampler": "temp=1.0", "policy": True},
    *[{"name": f"bon{n}", "ckpt": SFT, "sampler": "temp=1.0", "best_of": n} for n in (4, 16)],
    *[{"name": f"steer_a{a}", "ckpt": SFT, "sampler": "temp=1.0", "steer": {"vec": "runs/e2/steer_vec.pt", "layer": 4, "alpha": a}}
      for a in (2.0, 6.0)],
    {"name": "divpo", "ckpt": "runs/f1/divpo.pt", "sampler": "temp=1.0", "policy": True},
]


def bank_system(sysd: dict, headers: list[dict], n: int, tok, reward, max_new: int, smoke: bool) -> list[dict]:
    ckpt = None if smoke else sysd["ckpt"]
    model = load_model(ckpt, tok.vocab_size)
    g = torch.Generator(device="cuda").manual_seed(0)
    k = sysd.get("best_of", 1)
    ctx = contextlib.nullcontext()
    if "steer" in sysd:
        s = sysd["steer"]
        vec = torch.randn(model.cfg.d_model) if smoke else torch.load(s["vec"])
        ctx = steer(model, s["layer"], vec, s["alpha"])
    with ctx:
        rows = sample_set(model, headers, n * k, make_sampler(sysd["sampler"]), tok, max_new, generator=g,
                          instruction=sysd.get("instruction", ""))
    for r in rows:
        r["reward"] = reward(r["header"], r["poem"])
    if k > 1:                                                # best-of-k: no tractable log q. Keep the KL bound.
        rows = [max(rows[i : i + k], key=lambda r: r["reward"]) for i in range(0, len(rows), k)]
        for r in rows:
            r["logq"], r["kl_bound"] = None, math.log(k) - (k - 1) / k
    del model
    torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="runs/f1/systems.json")
    ap.add_argument("--tok", default="data/fineweb_tok.json")
    ap.add_argument("--reward", default=None)
    ap.add_argument("--headers", type=int, default=200)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--out", default="runs/f1/bank")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if a.smoke:
        a.headers, a.samples, a.max_new, out = 4, 4, 24, Path("runs/f1/bank_smoke")
        systems = [TEMPLATE[1], TEMPLATE[0], {**TEMPLATE[9], "best_of": 2}, TEMPLATE[11]]
    else:
        p = Path(a.systems)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(TEMPLATE, indent=1))
            print(f"wrote the template {p}. Edit the checkpoint paths, then run again.")
            return
        systems = json.loads(p.read_text())
    tok, reward = load_tokenizer(a.tok), load_reward(a.reward)
    poems = verse_split("val", a.headers)
    headers = [dict(m, lines=count_lines(t)) for m, t in poems]
    write_jsonl(out / "human.jsonl", [{"header": h, "poem": t} for h, (_, t) in zip(headers, poems)])
    for s in systems:
        f = out / f"{s['name']}.jsonl"
        if f.exists() or (not a.smoke and not Path(s["ckpt"]).exists()):
            print(f"skip {s['name']}: {'done' if f.exists() else 'no checkpoint ' + s['ckpt']}")
            continue
        rows = bank_system(s, headers, a.samples, tok, reward, a.max_new, a.smoke)
        write_jsonl(f, [{**r, "system": s["name"]} for r in rows])
        (out / f"{s['name']}.meta.json").write_text(json.dumps(s))
        print(f"{s['name']}: {len(rows)} samples, mean reward {sum(r['reward'] for r in rows) / len(rows):.3f}", flush=True)


if __name__ == "__main__":
    main()
