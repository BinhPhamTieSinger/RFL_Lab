"""Multi-Krum (Blanchard et al., 2017), ported from the authors' `MultiKrumAggregator`
(src/baselines/multi_krum_aggregator.py): score each update by the summed squared distance to its n - f - 2
nearest others, average the m lowest-scoring ones (plain mean, not sample-weighted). Two or fewer updates are
simply averaged.

The authors set f from the true number of attackers when there are any (an oracle no real server has), else
(n - 3) // 2. The registry reproduces that. Their distances are computed on whole model vectors, ours on updates;
distances are the same since every client starts from the same global model."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from rfl.core.types import ClientUpdate, StateDelta


def krum_scores(stacked: torch.Tensor, f: int) -> torch.Tensor:
    n = stacked.shape[0]
    k = max(1, n - min(f, max(0, (n - 3) // 2)) - 2)
    sq = (stacked ** 2).sum(dim=1)
    dists = (sq.unsqueeze(0) + sq.unsqueeze(1) - 2.0 * (stacked @ stacked.T)).clamp(min=0.0)
    dists.fill_diagonal_(float("inf"))
    return torch.topk(dists, k=min(k, n - 1), dim=1, largest=False).values.sum(dim=1)


class MultiKrum:
    def __init__(self, num_selected: int = 3, num_byzantine: int | None = None) -> None:
        self.m = num_selected
        self.f = num_byzantine
        self.last_selected: list[int] = []  # client ids chosen in the last call
        self.last_scores: dict[int, float] = {}  # client id -> Krum score in the last call (lower = more central)

    def aggregate(self, updates: Sequence[ClientUpdate], weights: Mapping[int, float] | None = None) -> StateDelta:
        if not updates:
            raise ValueError("Multi-Krum needs at least one update")
        keys = list(updates[0].delta)
        stacked = torch.stack([torch.cat([u.delta[k].reshape(-1) for k in keys]) for u in updates])
        n = len(updates)
        if n <= 2:
            chosen = list(range(n))
        else:
            f = self.f if self.f is not None else max(1, (n - 3) // 2)
            scores = krum_scores(stacked, f)
            self.last_scores = {u.client_id: float(s) for u, s in zip(updates, scores.tolist(), strict=True)}
            chosen = torch.topk(scores, k=max(1, min(self.m, n)), largest=False).indices.tolist()
        self.last_selected = [updates[i].client_id for i in chosen]
        mean = stacked[chosen].mean(dim=0)
        out, offset = {}, 0
        for k in keys:
            numel = updates[0].delta[k].numel()
            out[k] = mean[offset:offset + numel].view_as(updates[0].delta[k])
            offset += numel
        return out
