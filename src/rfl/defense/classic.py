"""Phase 5: established robust aggregation rules, for FedHEP's Global step (rfl.server.robust_fedhep_server).

Each rule takes the senders' weight updates as rows of a [n, D] tensor (BatchNorm buffers are left out by the
caller; Phase 2 showed their batch counters dominate distances) and returns a `RuleResult`: the aggregate update,
which rows it accepted (None for rules that keep everyone), and one score per row (higher = more trusted).

  * coordinate median and trimmed mean (Yin et al., ICML 2018): per coordinate, the median / the mean after
    dropping the k largest and k smallest values;
  * Bulyan (El Mhamdi et al., ICML 2018): select theta = n - 2f updates by repeatedly taking the Krum winner
    among those left (Krum score: summed squared distance to the r - f - 2 nearest others), then per coordinate
    average the beta = theta - 2f selected values closest to their median. Needs n >= 4f + 3;
  * FLAME (Nguyen et al., USENIX Security 2022): HDBSCAN on pairwise cosine distances of the local models
    (min cluster size n/2 + 1, so the admitted cluster is the majority), clip each admitted update to the median
    distance S of all local models from the global model, average, add N(0, (lambda S)^2) noise;
  * FLTrust (Cao et al., NDSS 2021): the server trains on a small clean root set; trust = ReLU(cos(update,
    server update)); each update is rescaled to the server update's norm; trust-weighted average;
  * ARC + NNM + trimmed mean (Phase 6, layer L0 of docs/RESEARCH_PLAN.md): adaptive robust clipping (Allouah et
    al., ICLR 2025) clips the k = floor(2 (f/n) (n - f)) largest rows to the norm of the (k+1)-th largest;
    nearest-neighbour mixing (Allouah et al., AISTATS 2023) replaces each row by the mean of its n - f nearest
    rows, itself included; then coordinate trimmed mean. The ARC paper does not fix ARC's place relative to
    NNM; clipping first keeps one large row from entering its neighbours' averages (DECISIONS.md, D31)."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from rfl.defense.soft_cosine import coordinate_median


@dataclass(frozen=True)
class RuleResult:
    update: torch.Tensor
    accepted: list[int] | None   # rows used; None = every row (median, trimmed mean)
    scores: torch.Tensor          # one per row, higher = more trusted


def _cos_to(stack: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(stack, ref.unsqueeze(0), dim=1, eps=1e-10)


def median_rule(stack: torch.Tensor) -> RuleResult:
    agg = coordinate_median(stack)
    return RuleResult(agg, None, _cos_to(stack, agg))


def trimmed_mean_rule(stack: torch.Tensor, k: int) -> RuleResult:
    n = stack.shape[0]
    if n <= 2 * k:
        raise ValueError(f"trimmed mean drops {k} per side, needs more than {2 * k} updates, got {n}")
    agg = torch.sort(stack, dim=0).values[k:n - k].mean(dim=0)
    return RuleResult(agg, None, _cos_to(stack, agg))


def _sq_dists(stack: torch.Tensor) -> torch.Tensor:
    """Pairwise squared distances, in float64 on the CPU (Apple's MPS has no float64)."""
    x = stack.detach().cpu().double()
    sq = (x ** 2).sum(dim=1)
    return (sq.unsqueeze(0) + sq.unsqueeze(1) - 2.0 * (x @ x.T)).clamp(min=0.0)


def bulyan_rule(stack: torch.Tensor, f: int) -> RuleResult:
    n = stack.shape[0]
    if n < 4 * f + 3:
        raise ValueError(f"Bulyan needs n >= 4f + 3, got n = {n}, f = {f}")
    d = _sq_dists(stack)
    left, chosen = list(range(n)), list[int]()
    first_scores = torch.zeros(n, dtype=torch.float64)
    while len(chosen) < n - 2 * f:
        sub = d[left][:, left].clone()
        sub.fill_diagonal_(float("inf"))
        k = max(1, len(left) - f - 2)
        scores = torch.topk(sub, k=min(k, len(left) - 1), dim=1, largest=False).values.sum(dim=1)
        if len(chosen) == 0:
            first_scores = scores
        best = left[int(torch.argmin(scores))]
        chosen.append(best)
        left.remove(best)
    sel = stack[chosen]
    beta = len(chosen) - 2 * f
    med = coordinate_median(sel)
    closest = torch.topk((sel - med).abs(), k=beta, dim=0, largest=False).indices
    agg = torch.gather(sel, 0, closest).mean(dim=0)
    return RuleResult(agg, sorted(chosen), -first_scores.float())


def flame_rule(stack: torch.Tensor, start: torch.Tensor, lam: float, generator: torch.Generator,
               on: str = "models") -> RuleResult:
    """`stack`: updates; `start`: the global weights they were trained from (same coordinates). Cosine
    distances on the local models start + update, as in the paper's equations (`on="models"`), or on the
    updates (`on="updates"`, the other reading of its wording)."""
    from sklearn.cluster import HDBSCAN

    n = stack.shape[0]
    x = stack.detach().cpu().double()  # float64 on the CPU: model cosines sit near 1, and MPS has no float64
    vecs = x + start.detach().cpu().double().unsqueeze(0) if on == "models" else x
    unit = F.normalize(vecs, dim=1, eps=1e-12)
    dist = (1.0 - unit @ unit.T).clamp(min=0.0).cpu().numpy()
    np.fill_diagonal(dist, 0.0)
    with warnings.catch_warnings():  # scikit-learn 1.9 warns about a future default it does not use here
        warnings.simplefilter("ignore", FutureWarning)
        labels = HDBSCAN(min_cluster_size=n // 2 + 1, min_samples=1, metric="precomputed",
                         allow_single_cluster=True).fit(dist).labels_
    accepted = [i for i in range(n) if labels[i] != -1] or list(range(n))  # all noise: keep everyone
    norms = stack.norm(dim=1)
    bound = norms.median()  # S_t: median over all n models, rejected ones included
    clipped = stack[accepted] * (bound / norms[accepted].clamp(min=1e-12)).clamp(max=1.0).unsqueeze(1)
    noise = torch.randn(stack.shape[1], generator=generator).to(stack.device) * (lam * bound)
    scores = torch.tensor([1.0 if labels[i] != -1 else 0.0 for i in range(n)])
    return RuleResult(clipped.mean(dim=0) + noise, accepted, scores)


def fltrust_rule(stack: torch.Tensor, server_update: torch.Tensor) -> RuleResult:
    trust = F.relu(_cos_to(stack, server_update))
    accepted = [i for i, t in enumerate(trust.tolist()) if t > 0]
    if not accepted:
        return RuleResult(torch.zeros_like(server_update), [], trust)
    scaled = stack * (server_update.norm() / stack.norm(dim=1).clamp(min=1e-12)).unsqueeze(1)
    agg = (trust.unsqueeze(1) * scaled).sum(dim=0) / trust.sum()
    return RuleResult(agg, accepted, trust)


def arc_clip(stack: torch.Tensor, f: int) -> torch.Tensor:
    """Adaptive robust clipping: the k = floor(2 (f/n) (n - f)) largest-norm rows are scaled down to the norm of
    the (k+1)-th largest; the others are unchanged. Ties are broken by row order."""
    n = stack.shape[0]
    k = int(2 * f * (n - f) // n) if n else 0
    if k == 0 or k >= n:
        return stack
    norms = stack.norm(dim=1)
    order = torch.argsort(norms, descending=True, stable=True)
    bound = norms[order[k]]
    scale = torch.ones_like(norms)
    big = order[:k]
    scale[big] = torch.where(norms[big] > 0, bound / norms[big], torch.ones_like(norms[big]))
    return stack * scale.unsqueeze(1)


def nnm(stack: torch.Tensor, f: int) -> torch.Tensor:
    """Nearest-neighbour mixing: each row becomes the mean of its n - f nearest rows (Euclidean), itself
    included."""
    n = stack.shape[0]
    if f == 0:
        return stack.mean(dim=0, keepdim=True).expand_as(stack).clone()
    near = torch.argsort(_sq_dists(stack), dim=1, stable=True)[:, :n - f].to(stack.device)
    return stack[near].mean(dim=1)


def arc_nnm_trimmed_mean_rule(stack: torch.Tensor, f: int) -> RuleResult:
    """L0 influence bound: ARC, then NNM, then coordinate trimmed mean dropping f per side."""
    mixed = nnm(arc_clip(stack, f), f)
    agg = trimmed_mean_rule(mixed, f).update
    return RuleResult(agg, None, _cos_to(stack, agg))
