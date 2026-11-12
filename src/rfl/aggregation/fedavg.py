"""FedAvg (McMahan et al., 2017): weighted mean of client deltas, weighted by sample count by default."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from rfl.core.types import ClientUpdate, StateDelta


class FedAvg:
    def aggregate(
        self, updates: Sequence[ClientUpdate], weights: Mapping[int, float] | None = None
    ) -> StateDelta:
        if not updates:
            raise ValueError("FedAvg needs at least one update")
        if weights is None:
            w = [float(u.num_samples) for u in updates]
        else:
            w = [float(weights.get(u.client_id, 0.0)) for u in updates]
        if any(x < 0 for x in w):
            raise ValueError("FedAvg weights must be non-negative")
        total = sum(w)
        if total <= 0:
            raise ValueError("FedAvg weights sum to zero; nothing to aggregate")
        keys = updates[0].delta.keys()
        return {
            k: torch.stack([(wi / total) * u.delta[k] for wi, u in zip(w, updates, strict=True)]).sum(dim=0)
            for k in keys
        }
