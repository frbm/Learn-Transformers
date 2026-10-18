# Transformers course — reference implementation

Solutions to the coding sections of the Notion course *Transformers*, written as one reusable library plus thin
per-module experiments. The target reader is an engineer who is levelling up to research-engineer standard.
So the code shows the habits as well as the answers: oracles, ratio-of-sums metrics, one forward pass for every
use, exact resume, and timing that synchronises before it reads the clock.

## Layout

```
src/lm/                  the library: import it, never copy it
  model/                 config, attention kernels and masks, RoPE, KV caches, layers, GPT, chunked loss,
                         μP init (mup), LoRA / DoRA / PiSSA (lora), NF4 (quant)
  tokenizer/             byte-level BPE (bpe), Unigram lattice and EM (unigram)
  data/                  uint16 token files (tokens), C2 corpus pipeline: MinHash, LSH, splits, Batches (corpus),
                         Trainer sources (sources), SFT format / pack / masks / losses (sft)
  train/                 Trainer (trainer), lr schedules (schedule), AdamW / Muon (optim), callbacks,
                         memory model (memory)
  infer/                 samplers (sampling), batched engine with BatchCache (generate), speculative decoding
  eval/                  bpb meters (metrics), diversity metrics (diversity), embeddings and k-means (embedding)
  analysis/              curvature (HVP, GGN, Lanczos, Hutchinson), scaling (FLOPs, law fits)
  interp/                hooks, circuits and patching, sparse autoencoders, toy models of superposition
  rl/                    tabular RL, estimators, bandits, soft RL, preference losses (prefs), policy helpers
  verse/                 the course's creative thread: verse corpus, rewards, shared loaders
  baselines/             byte n-gram (A1)
  utils/                 GPU timing and peak measurements (bench)
experiments/<module>/    one script per task, e.g. experiments/a3/train.py
tests/                   the course's given tests (test_<module>_*.py) and extra tests
docs/<MODULE>.md         per module: files, commands, measured results, answers, research-engineering tricks
```

Rule of thumb: code that two modules use goes in `src/lm`. Code that produces one plot stays in `experiments/`.

## Setup and commands

```bash
python -m pip install -e ".[dev]"
python -m pytest -q -m "not slow"
python -m experiments.a3.prep
python -m experiments.c1.train --name c1_smoke --steps 20 --eval_every 10
```

Run every script from the repository root as a module (`python -m experiments.<module>.<script>`). Data goes to
`data/`, logs, plots and checkpoints go to `runs/` and `checkpoints/`. All three are git-ignored.

## The model: one forward for every use

`GPT.forward(idx, cache=None, *, position_ids=None, doc_ids=None, last_only=False, **cache_kw)`:

| use | call | what changes |
|---|---|---|
| training, evaluation | `model(idx)`, `model.loss(idx, y, reduction, ce_chunk)` | causal mask; optional per-block checkpointing; chunked cross-entropy |
| packed SFT (D1) | `model(idx, doc_ids=d, position_ids=p)` | block-causal mask (dense, or flex on CUDA + Triton), RoPE per row |
| decoding (A3, A4) | `model(idx, cache=model.init_cache(B))` | `KVCache`: one length, mask offset = cache position |
| batched decoding (D4) | `model(idx, cache=BatchCache(...), valid=v, slot=s)` | left padding, per-row positions, rollback, CUDA-graph step |
| interpretability (E1, E2) | `run_with_cache(model, idx)`, `hooks(model, {...})` | identity HookPoints, built in; the pattern is built only when hooked |

A cache implements `begin(T, **kw) -> AttnPlan`, `update(layer, k, v)` and `end(T)`. The model has no other
cache-specific code. Every fast path is tested against `attention_core`, the plain definition.

## Conventions

- Comments: only when the reason is not obvious. They use ASD-STE100 Simplified Technical English.
- Checkpoints: `{"cfg": cfg.to_dict(), "model": state_dict}`. Load with `Config.from_dict`, which ignores unknown keys.
- Metrics: sums first, then one division (`BpbMeter`, `Trainer.evaluate`). Never a mean of per-batch means.
- Reductions (softmax, norms, losses) run in fp32 under bf16 autocast, and in fp64 for fp64 oracle tests.
- Compute budget for smoke runs: about 2 minutes. Full-scale defaults are in each script. Each `docs/` file says
  which numbers come from full runs and which from smoke runs.

## Status

| module | status |
|---|---|
| A1–A4 | done; full Tier 0 run in A3; A4 extrapolation smoke-run only |
| B1 | done; coordinate check and depth study full runs, LR sweep smoke-run only |
| B2 | done; Newton–Schulz batched by shape after Task 6; bake-off smoke-run only |
| B3 | done; curvature probe on the Trainer, per-θ caches; smoke runs only |
| B4 | done; bootstrap boundary bug and prediction-file guard fixed; smoke runs only |
| C1 | done; measured P = 66.7 TFLOP/s, β = 403 GB/s; predictor within 5 %; Tier 1 run smoke-tested (compiled 2.3× faster) |
| C2 | done; verse corpus built (5 602 poems, 2.5 M tokens) in data/verse.jsonl and data/verse_*.bin |
| D1 | done; SFT on the shared Trainer (pluggable objective); smoke runs only (needs the C1 Tier 1 checkpoint) |
| D2, D3, D4, R1, E1, E2, F1 | code and tests written, docs missing; E1 has one failing test |
| F2, G1, G2 | the Notion pages are blank |

Known debt, to fix when each module is finished:
- Three samplers: `lm.infer.generate.generate`, `lm.rl.policy.sample`, and the local `sample` in `experiments/d1/analysis.py`. Keep the engine.
- `lm.rl.policy` and `lm.verse.common` both load models and format verse headers. Merge them into `lm.verse`.
- `experiments/a3/train.py` keeps the course's minimal A3 loop on purpose. Every later module uses `lm.train.Trainer`.
