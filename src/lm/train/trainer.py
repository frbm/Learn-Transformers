"""The training loop for every module from C1 on (and the B-track sweeps).

bf16 autocast with fp32 master weights; AdamW, Muon + AdamW, or any list of optimisers you build; warm-up then
cosine or WSD on each group's own base lr; clipping; token-weighted gradient accumulation; optional torch.compile;
bpb evaluation on fixed windows; exact checkpoint and resume; MFU; a step or a wall-clock budget; callbacks.
"""
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from pathlib import Path

from lm.analysis.scaling import flops_per_token
from lm.data.corpus import Batches
from lm.model import GPT, Config
from lm.train.schedule import lr_mult


def mfu(flops_per_step: float, step_time_s: float, peak_flops: float) -> float:
    return flops_per_step / (step_time_s * peak_flops)


def compile_available() -> tuple[bool, str]:
    """Inductor needs Triton for GPU kernels. The Windows PyTorch wheels do not ship it."""
    try:
        from torch.utils._triton import has_triton

        if has_triton():
            return True, ""
    except ImportError:
        pass
    return False, "Triton is missing (native Windows). Install triton-windows or run in WSL2. Eager mode is used."


@dataclass
class TrainConfig:
    name: str = "c1_tier1"
    B: int = 16  # sequences per micro-batch
    T: int = 1024
    K: int = 1  # micro-batches per optimizer step
    steps: int = 12_207  # 200 M tokens at 16 x 1024
    budget_s: float | None = None  # wall-clock budget of training time. Then steps is only an upper bound.
    optimizer: str = "adamw"  # "adamw" or "muon". Ignored when the caller passes optimisers.
    lr: float = 3e-3  # AdamW
    muon_lr: float = 0.02
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    warmup: int = 200
    schedule: str = "cosine"
    min_lr_frac: float = 0.1
    clip: float = 1.0
    compile: bool = False
    ce_chunk: int | None = None
    eval_every: int = 250
    ckpt_every: int = 1000  # 0: never
    log_every: int = 1
    seed: int = 0
    device: str = "cuda"
    out_dir: str = "runs"
    peak_flops: float = 0.0  # 0: C1 Task 1's measured P (runs/c1_bench.json), else 71 TFLOP/s


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def default_optimizers(model: GPT, tc: TrainConfig) -> list[torch.optim.Optimizer]:
    from lm.train.optim import adamw, build_optimizers

    if tc.optimizer == "muon":
        return build_optimizers(model, tc.muon_lr, tc.lr, tc.weight_decay, betas=tc.betas)
    return [adamw(model, tc.lr, tc.weight_decay, tc.betas)]


class Trainer:
    def __init__(self, cfg: Config, tc: TrainConfig, train_tokens: np.ndarray, val: tuple[torch.Tensor, torch.Tensor],
                 bytes_per_token: float, *, model: GPT | None = None, opts: list | None = None,
                 callbacks: list | tuple = (), objective=None):
        """model and opts: pass them to use a custom init (μP), an optimiser, or a subset of parameters (LoRA).

        objective(trainer) -> list of zero-argument closures, one per micro-batch. Each returns its loss already
        divided by the step's denominator, so that the sum of the closures is the step's objective. None means
        next-token pretraining on train_tokens (token-weighted over the K micro-batches).
        """
        self.cfg, self.tc, self.bytes_per_token = cfg, tc, bytes_per_token
        self.dir = Path(tc.out_dir) / tc.name
        self.dir.mkdir(parents=True, exist_ok=True)
        seed_all(tc.seed)
        self.model = model if model is not None else GPT(cfg).to(tc.device)
        self.opts = opts if opts is not None else default_optimizers(self.model, tc)
        for opt in self.opts:
            for g in opt.param_groups:
                g.setdefault("base_lr", g["lr"])
        self.callbacks = list(callbacks)
        self.objective = objective or self._lm_objective
        self.batches = Batches(train_tokens, tc.B, tc.T, seed=tc.seed, device=tc.device)
        self.val = val
        self.step, self.tokens_seen, self.train_time = 0, 0, 0.0
        self.loss_fn = self.model.loss
        if tc.compile:
            ok, why = compile_available()
            if ok:
                self.loss_fn = torch.compile(self.model.loss)
            else:
                print(f"warning: --compile ignored. {why}")
        self.cuda = tc.device == "cuda"
        if not tc.peak_flops:
            from lm.utils.bench import load_peaks

            tc.peak_flops = load_peaks()[0]
        self.flops_per_step = flops_per_token(cfg, tc.T, training=True) * tc.B * tc.T * tc.K
        self.log_path = self.dir / "log.jsonl"

    @property
    def progress(self) -> float:
        return self.train_time / self.tc.budget_s if self.tc.budget_s else self.step / self.tc.steps

    @property
    def done(self) -> bool:
        return self.progress >= 1.0 or self.step >= self.tc.steps

    def autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.cuda)

    def train_step(self) -> dict:
        """One optimizer step over K micro-batches. Returns GPU scalars: no host sync here."""
        tc = self.tc
        mult = lr_mult(self.step, tc.steps, tc.warmup, tc.schedule, min_frac=tc.min_lr_frac,
                       progress=self.progress if tc.budget_s else None)
        for opt in self.opts:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * mult
        total = torch.zeros((), device=tc.device)
        for closure in self.objective(self):
            with self.autocast():
                loss = closure()
            loss.backward()
            total += loss.detach()
        gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc.clip if tc.clip else float("inf"))
        for opt in self.opts:
            opt.step()
        for opt in self.opts:
            opt.zero_grad(set_to_none=True)
        self.step += 1
        self.tokens_seen += tc.K * tc.B * tc.T
        return {"loss": total, "grad_norm": gnorm, "lr": mult * self.opts[-1].param_groups[0]["base_lr"]}

    def _lm_objective(self, _=None) -> list:
        micro = [next(self.batches) for _ in range(self.tc.K)]
        # Count the loss-bearing tokens of the whole step before any backward (C1 Proposition 8).
        n_tok = sum((y != -100).sum() for _, y in micro)
        return [lambda x=x, y=y: self.loss_fn(x, y, reduction="sum", ce_chunk=self.tc.ce_chunk) / n_tok
                for x, y in micro]

    @torch.no_grad()
    def evaluate(self, batch_size: int = 8) -> dict:
        """Mean loss over the fixed held-out windows as a ratio of sums, then bits per byte."""
        self.model.eval()
        x, y = self.val
        total, n = torch.zeros((), device=x.device, dtype=torch.float64), 0
        for i in range(0, len(x), batch_size):
            with self.autocast():
                total += self.model.loss(x[i : i + batch_size], y[i : i + batch_size], reduction="sum").double()
            n += int((y[i : i + batch_size] != -100).sum())
        self.model.train()
        loss = total.item() / n
        return {"val_loss": loss, "val_bpb": loss / (math.log(2) * self.bytes_per_token)}

    def state_dict(self) -> dict:
        s = {"step": self.step, "tokens_seen": self.tokens_seen, "train_time": self.train_time,
             "model": self.model.state_dict(), "opts": [o.state_dict() for o in self.opts],
             "batches": self.batches.state_dict(), "cfg": asdict(self.cfg), "tc": asdict(self.tc),
             "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}}
        if torch.cuda.is_available():
            s["rng"]["cuda"] = torch.cuda.get_rng_state_all()
        return s

    def save(self, path: Path | None = None) -> Path:
        path = path or self.dir / "ckpt.pt"
        tmp = path.with_suffix(".tmp")
        torch.save(self.state_dict(), tmp)
        os.replace(tmp, path)  # Atomic: a crash during save keeps the previous checkpoint.
        return path

    def load(self, path: Path | None = None) -> None:
        s = torch.load(path or self.dir / "ckpt.pt", map_location=self.tc.device, weights_only=False)
        self.model.load_state_dict(s["model"])
        for o, os_ in zip(self.opts, s["opts"]):
            o.load_state_dict(os_)
            for g in o.param_groups:  # load_state_dict restores lr. base_lr must come from this run.
                g.setdefault("base_lr", g["lr"])
        self.batches.load_state_dict(s["batches"])
        self.step, self.tokens_seen, self.train_time = s["step"], s["tokens_seen"], s.get("train_time", 0.0)
        random.setstate(s["rng"]["python"])
        np.random.set_state(s["rng"]["numpy"])
        torch.set_rng_state(s["rng"]["torch"].cpu())
        if "cuda" in s["rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([t.cpu() for t in s["rng"]["cuda"]])

    def log(self, rec: dict) -> None:
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _sync(self) -> None:
        if self.cuda:
            torch.cuda.synchronize()

    def _eval_record(self, train_loss: float) -> dict:
        ev = {"step": self.step, "tokens": self.tokens_seen, "time_s": self.train_time, "train_loss": train_loss,
              **self.evaluate()}
        self.log(ev)
        return ev

    def fit(self, until: int | None = None, verbose: bool = True) -> list[dict]:
        """Train until step `until`, tc.steps or tc.budget_s. Stop early when the loss is not finite.

        Training time excludes evaluation, checkpoints, callbacks and step 0 (kernel selection, optimiser state).
        """
        tc = self.tc
        until = until or tc.steps
        recs, pending, last_loss = [], [], float("nan")
        say = (lambda r: print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}),
                               flush=True)) if verbose else (lambda r: None)
        # A budget needs a precise clock. Then synchronise every step. Otherwise only on logged steps.
        sync_each = tc.budget_s is not None
        if self.cuda:
            torch.cuda.reset_peak_memory_stats()
        while self.step < until and not self.done:
            if self.step % tc.eval_every == 0:
                recs.append(self._eval_record(last_loss)); say(recs[-1])
                if not math.isfinite(recs[-1]["val_loss"]):
                    return recs
            for cb in self.callbacks:
                cb.before_step(self)
            if not pending:
                self._sync()
                t0 = time.perf_counter()
            pending.append(self.train_step())
            # Close the timing window before anything that is not training: a log, an eval, a callback, a save.
            logged = self.step % tc.log_every == 0 or self.step == until
            ckpt = tc.ckpt_every and self.step % tc.ckpt_every == 0
            if sync_each or logged or ckpt or self.callbacks or self.step % tc.eval_every == 0:
                self._sync()
                n, dt = len(pending), time.perf_counter() - t0
                first = self.step - n == 0
                self.train_time += dt * (n - 1) / n if first else dt
                last_loss = pending[-1]["loss"].item()
                if logged:
                    last = pending[-1]
                    rec = {"step": self.step, "tokens": self.tokens_seen, "loss": last_loss, "lr": last["lr"],
                           "grad_norm": last["grad_norm"].item(), "step_time": dt / n,
                           "tok_per_s": tc.K * tc.B * tc.T * n / dt, "mfu": mfu(self.flops_per_step, dt / n, tc.peak_flops),
                           "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30 if self.cuda else 0.0}
                    self.log(rec); recs.append(rec)
                    if self.step % max(1, tc.eval_every // 5) == 0 or self.step == until:
                        say(rec)
                pending = []
                if not math.isfinite(last_loss):
                    break
            for cb in self.callbacks:
                cb.after_step(self)
            if ckpt:
                self.save()
        if self.done or not math.isfinite(last_loss):
            recs.append(self._eval_record(last_loss)); say(recs[-1])
        return recs


def train_run(model: GPT, opts: list, data, val: tuple, *, B: int, T: int, steps: int | None = None,
              budget_s: float | None = None, warmup: int = 40, eval_every: int = 100, seed: int = 0,
              clip: float = 1.0, track_every: int = 0, name: str = "sweep", verbose: bool = False):
    """One sweep run (B1, B2, B4): a prebuilt model and optimisers, a step count or a wall-clock budget.

    data: a lm.data.tokens.TokenData. Returns (one record per evaluation, B2's update records).
    The schedule is warm-up then cosine to 0 over steps or over the budget.
    """
    from lm.train.callbacks import UpdateTracker

    assert (steps is None) != (budget_s is None), "give steps or budget_s"
    tc = TrainConfig(name=name, B=B, T=T, steps=steps or 10**9, budget_s=budget_s, warmup=warmup, min_lr_frac=0.0,
                     clip=clip, eval_every=eval_every, ckpt_every=0, log_every=eval_every, seed=seed,
                     out_dir="runs/sweeps", device=str(model.device.type))
    tracker = [UpdateTracker(track_every)] if track_every else []
    tr = Trainer(model.cfg, tc, data.data[: data.split], val, data.bytes_per_token, model=model, opts=opts,
                 callbacks=tracker)
    evals = [r for r in tr.fit(verbose=verbose) if "val_loss" in r]
    return evals, (tracker[0].records if tracker else [])
