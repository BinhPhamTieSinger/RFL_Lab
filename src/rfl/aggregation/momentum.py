"""Server-side momentum for a parent, ported from the authors' `_apply_cluster_momentum`
(src/core/hierarchical_ensemble_engine.py, with `cluster_momentum_beta = 0`, their default).

new_parent = beta * old_parent + (1 - beta) * aggregate, with beta = V / (V + ||mean update||^2), V the mean
squared distance of members' updates from their mean, clipped to [0.1, 0.9].
A group that disagrees moves slowly."""
from __future__ import annotations

from collections.abc import Sequence

import torch


def adaptive_momentum_beta(member_deltas: Sequence[torch.Tensor]) -> float:
    """0 (no momentum) for groups with fewer than two members, as in the authors' code."""
    if len(member_deltas) < 2:
        return 0.0
    stacked = torch.stack(list(member_deltas))
    mean = stacked.mean(dim=0)
    norm_sq = torch.sum(mean ** 2).item()
    var = torch.mean(torch.sum((stacked - mean) ** 2, dim=1)).item()
    return min(0.90, max(0.10, float(var / (var + norm_sq + 1e-8))))
