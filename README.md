# Transformers course — reference implementation

Solutions to the coding sections of *Transformers* on my Notion.
Written as a reusable library.

## Layout

```
src/lm/                  library level
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
  verse/                 creative writing part of the course: verse corpus, rewards, shared loaders
  baselines/             byte n-gram (A1)
  utils/                 GPU timing and peak measurements (bench)
experiments/<module>/    one script per task, e.g. experiments/a3/train.py
tests/                   tests from the course (test_<module>_*.py) plus extra
docs/<MODULE>.md         per module: files, commands, measured results, answers, research-engineering tricks to remember
```

Rule of thumb: code used by 2+ modules goes in `src/lm`. Code that produces one plot stays in `experiments/`.

## Setup and commands

```bash
python -m pip install -e ".[dev]"
python -m pytest -q -m "not slow"
python -m experiments.a3.prep
python -m experiments.c1.train --name c1_smoke --steps 20 --eval_every 10
```

- Scripts should be run from the repo root as a module (`python -m experiments.<module>.<script>`).
- Data goes to `data/`.
- Logs, plots and checkpoints go to `runs/` and `checkpoints/`.
- NB: all three are git-ignored.

## Main model

`GPT.forward(idx, cache=None, *, position_ids=None, doc_ids=None, last_only=False, **cache_kw)`:

| use | call |
|---|---|
| training, evaluation | `model(idx)`, `model.loss(idx, y, reduction, ce_chunk)` |
| packed SFT (D1) | `model(idx, doc_ids=d, position_ids=p)` |
| decoding (A3, A4) | `model(idx, cache=model.init_cache(B))` |
| batched decoding (D4) | `model(idx, cache=BatchCache(...), valid=v, slot=s)` |
| interpretability (E1, E2) | `run_with_cache(model, idx)`, `hooks(model, {...})` |

## Conventions

- Checkpoints: `{"cfg": cfg.to_dict(), "model": state_dict}`. Load with `Config.from_dict`, which ignores unknown keys.
- Metrics: sums then one division.
- Reductions (softmax, norms, losses): fp32 under bf16 autocast, or fp64 for fp64 oracle tests.

## Status

| module | status |
|---|---|
| A1-A4 | done |
| B1-B4 | done |
| C1-C2 | done |
| D1 | done |
| D2, D3, D4, R1, E1, E2, F1 | code and tests written, docs missing; E1 has one failing test |
| F2, G1, G2 | Notion in progress |
