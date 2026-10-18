"""D1 on the shared Trainer: the SFT objective with replay, and the SFT evaluation callback."""
import json

import numpy as np
import torch

from lm.data.sft import entropy_from_logits, loss_denominator, response_rel_pos, sft_loss_sums

N_BINS = 20


def _rows(packed: dict, idx) -> dict:
    return {k: packed[k][idx] for k in ("input_ids", "labels", "doc_ids", "position_ids")}


class SFTObjective:
    """K micro-batches per step: round(lam K) pretraining batches, the rest packed SFT rows.

    The objective is (1 - lam) L_sft + lam L_pre (Proposition 5), with lam the realised fraction round(lam K) / K.
    L_sft is token- or sequence-weighted over the whole step (C1 Proposition 8, one level up). An epoch is one
    pass over a random permutation of the packed rows.
    """

    def __init__(self, packed: dict, B: int, K: int, lam: float = 0.0, weighting: str = "token",
                 pre_batches=None, flex: bool | None = None, seed: int = 0):
        self.p, self.B, self.weighting, self.flex = packed, B, weighting, flex
        self.K_pre = round(lam * K)
        self.K_sft = K - self.K_pre
        self.lam = self.K_pre / K
        assert self.K_sft > 0, "at least one SFT micro-batch per step"
        self.pre = pre_batches  # None: the Trainer's own token stream (its train_tokens)
        self.rng = np.random.default_rng(seed)
        self.R = len(packed["input_ids"])
        self.steps_per_epoch = max(1, self.R // (self.K_sft * B))
        self.perm, self.cursor = self.rng.permutation(self.R), 0

    def __call__(self, tr) -> list:
        n = self.K_sft * self.B
        if self.cursor + n > self.R:
            self.perm, self.cursor = self.rng.permutation(self.R), 0
        idx = torch.from_numpy(self.perm[self.cursor : self.cursor + n])
        self.cursor += n
        den = loss_denominator(self.p["labels"][idx], self.p["doc_ids"][idx], self.weighting)
        dev = tr.model.device
        closures = []
        for j in range(self.K_sft):
            g = {k: v.to(dev, non_blocking=True) for k, v in _rows(self.p, idx[j * self.B : (j + 1) * self.B]).items()}

            def sft(g=g):
                logits = tr.model(g["input_ids"], doc_ids=g["doc_ids"], position_ids=g["position_ids"], flex=self.flex)
                return (1 - self.lam) * sft_loss_sums(logits, g["labels"], g["doc_ids"], self.weighting)[0] / den
            closures.append(sft)
        for _ in range(self.K_pre):
            x, y = next(self.pre or tr.batches)
            closures.append(lambda x=x, y=y: self.lam / self.K_pre * tr.model.loss(x, y))
        return closures


@torch.no_grad()
def evaluate_sft(model, packed: dict, B: int = 8) -> dict:
    """Response-token loss (token mean, ratio of sums), mean entropy, and entropy per relative position."""
    was = model.training
    model.eval()
    dev = model.device
    nll = ent = cnt = 0.0
    bins_s = torch.zeros(N_BINS, dtype=torch.float64)
    bins_n = torch.zeros(N_BINS, dtype=torch.float64)
    for i in range(0, len(packed["input_ids"]), B):
        g = {k: v.to(dev) for k, v in _rows(packed, slice(i, i + B)).items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            logits = model(g["input_ids"], doc_ids=g["doc_ids"], position_ids=g["position_ids"], flex=False)
        m = g["labels"] != -100
        nll += sft_loss_sums(logits, g["labels"], g["doc_ids"], "token")[0].item()
        h = entropy_from_logits(logits)[m].double()
        ent += h.sum().item()
        cnt += int(m.sum())
        bi = (response_rel_pos(g["labels"], g["doc_ids"])[m] * N_BINS).long().clamp(max=N_BINS - 1)
        bins_s.index_add_(0, bi.cpu(), h.cpu())
        bins_n.index_add_(0, bi.cpu(), torch.ones_like(h).cpu())
    model.train(was)
    return {"loss": nll / cnt, "entropy": ent / cnt, "entropy_bins": (bins_s / bins_n.clamp(min=1)).tolist()}


class SFTEval:
    """Trainer callback. Every `every` steps (and at step 0): verse train and val loss on response tokens,
    FineWeb bpb (the Trainer's own validation windows: forgetting), mean entropy and entropy by position."""

    def __init__(self, val: dict, train_sample: dict, every: int, tag: str = "", extra: dict | None = None):
        self.val, self.train_sample, self.every, self.tag, self.extra = val, train_sample, every, tag, extra or {}
        self.records: list[dict] = []

    def _record(self, tr) -> None:
        v, t = evaluate_sft(tr.model, self.val), evaluate_sft(tr.model, self.train_sample)
        rec = {"tag": self.tag, "step": tr.step, "train_loss": t["loss"], "val_loss": v["loss"],
               "fineweb_bpb": tr.evaluate()["val_bpb"], "entropy": v["entropy"], "entropy_bins": v["entropy_bins"],
               **self.extra}
        self.records.append(rec)
        tr.log({"sft_eval": rec})
        print(json.dumps({k: round(x, 4) if isinstance(x, float) else x for k, x in rec.items()
                          if k != "entropy_bins"}), flush=True)

    def before_step(self, tr) -> None:
        if tr.step == 0:
            self._record(tr)

    def after_step(self, tr) -> None:
        if tr.step % self.every == 0 or tr.step == tr.tc.steps:
            self._record(tr)


def train_sample(packed: dict, n: int) -> dict:
    """n evenly spaced training rows: the train loss on the same number of rows as validation."""
    sel = torch.from_numpy(np.linspace(0, len(packed["input_ids"]) - 1, min(n, len(packed["input_ids"]))).astype(np.int64))
    return _rows(packed, sel)
