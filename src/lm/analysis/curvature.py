"""B3. Matrix-free curvature: HVP, GGN-vector product, Lanczos, Hutchinson. Flat parameter vectors everywhere."""
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.func import functional_call, jvp, vjp

Matvec = Callable[[Tensor], Tensor]


def hvp(loss_fn: Callable[[Tensor], Tensor], theta: Tensor, v: Tensor) -> Tensor:
    """H v by double backward (Pearlmutter). autograd.grad never touches .grad buffers."""
    theta = theta.detach().requires_grad_(True)
    (g,) = torch.autograd.grad(loss_fn(theta), theta, create_graph=True)
    (hv,) = torch.autograd.grad(g @ v, theta)
    return hv


def upcast(t: Tensor) -> Tensor:
    """bf16 and fp16 go to fp32. float64 stays float64."""
    return t.to(torch.promote_types(t.dtype, torch.float32))


def softmax_hessian_vp(logits: Tensor, u: Tensor) -> Tensor:
    """Lambda u = p * u - p <p, u>, row-wise. Lambda = diag(p) - p p^T does not depend on the label."""
    p, u = upcast(logits).softmax(-1), upcast(u)
    return (p * u - p * (p * u).sum(-1, keepdim=True)).to(logits.dtype)


def ggn_vp(logits_fn: Callable[[Tensor], Tensor], theta: Tensor, v: Tensor) -> Tensor:
    """G v = (1/N) J^T Lambda J v: one JVP (forward mode), Lambda in closed form, one VJP."""
    theta = theta.detach()
    logits, jv = jvp(logits_fn, (theta,), (v,))
    _, vjp_fn = vjp(logits_fn, theta)
    n = logits.numel() // logits.shape[-1]
    return vjp_fn(softmax_hessian_vp(logits, jv))[0] / n


def lanczos(matvec: Matvec, dim: int, k: int, reorth: bool = True, *, device=None, dtype=torch.float64,
            seed: int = 0, v0: Tensor | None = None) -> Tensor:
    """Ritz values of a symmetric operator after k matvecs, sorted descending."""
    k = min(k, dim)
    gen = torch.Generator(device or "cpu").manual_seed(seed)
    q = torch.randn(dim, device=device, dtype=dtype, generator=gen) if v0 is None else v0.to(device, dtype)
    q = q / q.norm()
    Q = torch.empty(k, dim, device=device, dtype=dtype)
    alpha, beta = [], []
    for j in range(k):
        Q[j] = q
        w = matvec(q).to(dtype)
        alpha.append(torch.dot(w, q))
        w = w - alpha[-1] * q - (beta[-1] * Q[j - 1] if j > 0 else 0)
        if reorth:
            # Classical Gram-Schmidt twice is as stable as modified Gram-Schmidt. It also stays in matmuls.
            for _ in range(2):
                w = w - Q[: j + 1].T @ (Q[: j + 1] @ w)
        b = w.norm()
        if j == k - 1 or b < 1e-10 * alpha[0].abs().clamp_min(1e-30):
            break
        beta.append(b)
        q = w / b
    a = torch.stack(alpha)
    T = torch.diag(a)
    if beta:
        bt = torch.stack(beta[: len(alpha) - 1])
        T = T + torch.diag(bt, 1) + torch.diag(bt, -1)
    return torch.linalg.eigvalsh(T.double()).flip(0)


def hutchinson_trace(matvec: Matvec, dim: int, n_samples: int, seed: int = 0, *, device=None,
                     dtype=torch.float64) -> float:
    """Mean of z^T A z over Rademacher probes z. Accumulate in float64 on the device: one host sync, at the end."""
    gen = torch.Generator(device or "cpu").manual_seed(seed)
    total = torch.zeros((), device=device, dtype=torch.float64)
    for _ in range(n_samples):
        z = torch.randint(0, 2, (dim,), device=device, generator=gen).to(dtype) * 2 - 1
        total += torch.dot(z, matvec(z).to(dtype)).double()
    return (total / n_samples).item()


# Thin wrappers for a real nn.Module. A flat vector maps to a dict of views. functional_call keeps tied weights.

class FlatModel:
    def __init__(self, model: nn.Module):
        self.model = model
        self.names, self.shapes, self.sizes = [], [], []
        for name, p in model.named_parameters():
            self.names.append(name); self.shapes.append(p.shape); self.sizes.append(p.numel())
        self.dim = sum(self.sizes)

    def flat(self) -> Tensor:
        return torch.cat([p.detach().float().flatten() for p in self.model.parameters()])

    def unflat(self, theta: Tensor) -> dict[str, Tensor]:
        return {n: t.view(s) for n, t, s in zip(self.names, theta.split(self.sizes), self.shapes)}

    def flat_state(self, opt: torch.optim.Optimizer, key: str) -> Tensor | None:
        """One optimiser state tensor, flattened in parameter order. None if a parameter lacks it."""
        out = []
        for p in self.model.parameters():
            s = opt.state.get(p, {})
            if key not in s:
                return None
            out.append(s[key].detach().float().flatten())
        return torch.cat(out)

    def logits_fn(self, x: Tensor) -> Callable[[Tensor], Tensor]:
        return lambda theta: functional_call(self.model, self.unflat(theta), (x,)).flatten(0, -2)

    def loss_fn(self, x: Tensor, y: Tensor) -> Callable[[Tensor], Tensor]:
        f = self.logits_fn(x)
        return lambda theta: F.cross_entropy(upcast(f(theta)), y.flatten())


class Curvature:
    """H v and G v on a fixed batch, split into chunks. Chunk results sum with weight n_chunk / N (ratio of sums).

    cache=True: theta is fixed for a whole measurement, so build the per-chunk structures once and reuse them for
    every matvec. HVP: keep the graph of the gradient (create_graph), then each product is one backward of
    <g, v>. GGN: keep the graph of the logits and of J^T u, then each product is two backward passes, with no
    new forward pass. The cost is memory: one graph per chunk, alive until rebind or release.
    Use it only when every chunk's graph fits at once. On Windows (WDDM), an overflow does not raise an
    out-of-memory error: the driver spills to system RAM and the job runs 10-100x slower.
    """

    def __init__(self, fm: FlatModel, x: Tensor, y: Tensor, chunk: int, cache: bool = False):
        self.fm, self.cache = fm, cache
        self.chunks = [(x[i:i + chunk], y[i:i + chunk]) for i in range(0, len(x), chunk)]
        self.weights = [xc.shape[0] / x.shape[0] for xc, _ in self.chunks]
        self.release()

    def release(self) -> None:
        self._key, self._grads, self._lins = None, None, None

    def _bind(self, theta: Tensor) -> None:
        # Key on identity and version. An in-place update (theta.sub_) keeps the identity but bumps the version.
        key = (id(theta), theta._version)
        if self._key != key:
            self.release()
            self._key = key

    def _sum(self, f: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
        return sum(w * f(xc, yc) for w, (xc, yc) in zip(self.weights, self.chunks))

    def value_and_grad(self, theta: Tensor) -> tuple[float, Tensor]:
        loss, grad = torch.zeros((), device=theta.device, dtype=theta.dtype), torch.zeros_like(theta)
        for w, (xc, yc) in zip(self.weights, self.chunks):
            t = theta.detach().requires_grad_(True)
            l = self.fm.loss_fn(xc, yc)(t)
            grad.add_(torch.autograd.grad(l, t)[0], alpha=w)
            loss += w * l.detach()
        return loss.item(), grad

    def hvp(self, theta: Tensor, v: Tensor) -> Tensor:
        if not self.cache:
            return self._sum(lambda xc, yc: hvp(self.fm.loss_fn(xc, yc), theta, v))
        self._bind(theta)
        if self._grads is None:
            self._grads = []
            for xc, yc in self.chunks:
                t = theta.detach().requires_grad_(True)
                (g,) = torch.autograd.grad(self.fm.loss_fn(xc, yc)(t), t, create_graph=True)
                self._grads.append((t, g))
        return sum(w * torch.autograd.grad(g @ v, t, retain_graph=True)[0]
                   for w, (t, g) in zip(self.weights, self._grads))

    def ggn_vp(self, theta: Tensor, v: Tensor) -> Tensor:
        if not self.cache:
            return self._sum(lambda xc, yc: ggn_vp(self.fm.logits_fn(xc), theta, v))
        self._bind(theta)
        if self._lins is None:
            # One forward per chunk. Jv by the double-backward trick: Jv = d<J^T u, v>/du for a dummy u.
            # It needs only reverse mode, so the graph of the logits is built once and reused.
            self._lins = []
            for xc, _ in self.chunks:
                t = theta.detach().requires_grad_(True)
                logits = self.fm.logits_fn(xc)(t)
                u = torch.zeros_like(logits, requires_grad=True)
                (jt_u,) = torch.autograd.grad(logits, t, grad_outputs=u, create_graph=True)
                self._lins.append((t, logits, u, jt_u))
        out = torch.zeros_like(theta)
        for w, (t, logits, u, jt_u) in zip(self.weights, self._lins):
            (jv,) = torch.autograd.grad(jt_u, u, grad_outputs=v, retain_graph=True)
            (jt_lam_jv,) = torch.autograd.grad(logits, t, grad_outputs=softmax_hessian_vp(logits.detach(), jv),
                                               retain_graph=True)
            n = logits.numel() // logits.shape[-1]
            out.add_(jt_lam_jv, alpha=w / n)
        return out


def preconditioned(matvec: Matvec, p_inv_sqrt: Tensor) -> Matvec:
    """v -> P^{-1/2} A P^{-1/2} v. Symmetric, with the same spectrum as P^{-1} A."""
    return lambda v: p_inv_sqrt * matvec(p_inv_sqrt * v)


def top_eigs(matvec: Matvec, dim: int, k: int, n_top: int, device, seed: int = 0) -> list[float]:
    """Top Ritz values in float32 storage. k vectors of size P are the only memory cost."""
    return lanczos(lambda v: matvec(v.float()), dim, k, device=device, dtype=torch.float32, seed=seed)[:n_top].tolist()


def adam_p_inv_sqrt(fm: FlatModel, opts) -> Tensor | None:
    """P^{-1/2} with P = diag(sqrt(v_hat) + eps), read from the AdamW state. None if a tensor is not on AdamW."""
    opt = next((o for o in opts if isinstance(o, torch.optim.AdamW)), None)
    v = fm.flat_state(opt, "exp_avg_sq") if opt is not None else None
    if v is None:
        return None
    g = opt.param_groups[0]
    t = float(opt.state[next(iter(fm.model.parameters()))]["step"])
    v_hat = v / (1 - g["betas"][1] ** t)
    return (v_hat.sqrt() + g["eps"]).rsqrt()


class CurvatureProbe:
    """Trainer callback (B3 Task 5). Every `every` steps, on a fixed batch, in fp32 and eval mode:
    top eigenvalues of H and G (Lanczos), tr G (Hutchinson), the preconditioned sharpness under AdamW,
    and the sharpness along the step actually taken, u^T H u / (-g^T u). The quadratic model decreases iff it is < 2.

    Every product goes through autograd.grad on a flat copy. The .grad buffers and the optimiser state stay intact.
    """

    def __init__(self, x: Tensor, y: Tensor, every: int, k: int = 10, n_top: int = 10, hutch: int = 5,
                 chunk: int = 2):
        self.x, self.y, self.every, self.k, self.n_top, self.hutch, self.chunk = x, y, every, k, n_top, hutch, chunk
        self.records: list[dict] = []
        self.time_s = 0.0
        self._pending = None

    def _timed(self, rec: dict, key: str, fn):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        out = fn()
        b.record()
        torch.cuda.synchronize()
        rec[f"t_{key}"] = a.elapsed_time(b) / 1e3
        self.time_s += rec[f"t_{key}"]
        return out

    def _math_sdpa(self):
        from torch.nn.attention import SDPBackend, sdpa_kernel
        # Double backward and forward-mode AD need the math kernel. Fused kernels support neither.
        return sdpa_kernel(SDPBackend.MATH)

    def before_step(self, tr) -> None:
        if tr.step % self.every:
            return
        model, dev = tr.model, tr.model.device
        fm = getattr(self, "fm", None) or FlatModel(model)
        self.fm = fm
        curv = Curvature(fm, self.x, self.y, self.chunk, cache=True)
        rec = {"step": tr.step}
        model.eval()
        theta = fm.flat()
        H, G = (lambda v: curv.hvp(theta, v)), (lambda v: curv.ggn_vp(theta, v))
        with self._math_sdpa():
            loss, grad = self._timed(rec, "grad", lambda: curv.value_and_grad(theta))
            rec["curv_loss"] = loss
            rec["eig_H"] = self._timed(rec, "eig_H", lambda: top_eigs(H, fm.dim, self.k, self.n_top, dev))
            rec["eig_G"] = self._timed(rec, "eig_G", lambda: top_eigs(G, fm.dim, self.k, self.n_top, dev))
            rec["tr_G"] = self._timed(rec, "tr_G", lambda: hutchinson_trace(G, fm.dim, self.hutch, device=dev,
                                                                            dtype=torch.float32))
            p = adam_p_inv_sqrt(fm, tr.opts)
            if p is not None:
                rec["precond_sharp"] = self._timed(rec, "precond", lambda: top_eigs(preconditioned(H, p), fm.dim,
                                                                                    self.k, 1, dev)[0])
        model.train()
        curv.release()
        self._pending = (curv, theta, grad, rec)

    def after_step(self, tr) -> None:
        if self._pending is None:
            return
        curv, theta, grad, rec = self._pending
        self._pending = None
        u = self.fm.flat() - theta
        with self._math_sdpa():
            uhu = self._timed(rec, "uhu", lambda: torch.dot(u, curv.hvp(theta, u)).item())
        curv.release()
        g_u = torch.dot(grad, u).item()
        rec.update(update_ratio=uhu / max(-g_u, 1e-30), u_norm=u.norm().item(), u_hu=uhu, g_u=g_u,
                   lrs=[o.param_groups[0]["lr"] for o in tr.opts], train_time=tr.train_time, probe_time=self.time_s)
        self.records.append(rec)
        tr.log({"curvature": rec})
