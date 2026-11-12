"""Coordinated update attacks, applied after local training in FedHEP servers.

ALIE: mean - z * coordinate std. Min-Max: mean - gamma * std, with largest
feasible gamma such that every distance to an observed update is bounded by
its population's maximum pairwise distance (Shejwalkar & Houmansadr, NDSS 2021).
Statistics use colluders' clean updates by default; omniscient mode uses honest
updates. Neither attack promises evasion of this defense's cosine/EMA threshold.
BatchNorm buffers are preserved from each attacker's clean upload.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch

from rfl.attacks.byzantine import ByzantineAttack


def crafted_update(observed: torch.Tensor, kind: str, strength: float = 1.0) -> torch.Tensor:
    mean = observed.mean(0)
    std = observed.std(0, correction=0)
    if kind == "alie":
        return mean - strength * std
    # squared distances avoid allocating an n x n x dimension tensor
    radius = max(float(((row - observed) ** 2).sum(1).max()) for row in observed)

    def feasible(gamma: float) -> bool:
        return float(((mean - gamma * std - observed) ** 2).sum(1).max()) <= radius

    lo, hi = 0.0, 1.0
    while feasible(hi) and hi < 1e6:
        lo, hi = hi, hi * 2
    for _ in range(40):
        mid = (lo + hi) / 2
        if feasible(mid):
            lo = mid
        else:
            hi = mid
    return mean - lo * std


class CoordinatedAttack(ByzantineAttack):
    def __init__(self, kind: str, strength: float = 1.0, knowledge: str = "colluders") -> None:
        super().__init__("label_flip")
        if kind not in ("alie", "min_max") or knowledge not in ("colluders", "omniscient"):
            raise ValueError("Unknown coordinated attack or knowledge model")
        self.kind, self.strength, self.knowledge = kind, strength, knowledge

    def poison_batch(self, uploads: Mapping[int, torch.Tensor], start: torch.Tensor,
                     attackers: frozenset[int], parameter_mask: torch.Tensor) -> dict[int, torch.Tensor]:
        result = dict(uploads)
        if not attackers:
            return result
        rows = sorted(attackers if self.knowledge == "colluders" else set(uploads) - attackers)
        if not rows:
            raise ValueError("Coordinated attack needs observed updates")
        mask = parameter_mask.to(start.device)
        observed = torch.stack([(uploads[c] - start)[mask] for c in rows])
        delta = crafted_update(observed, self.kind, self.strength)
        for c in attackers:
            poisoned = uploads[c].clone()
            poisoned[mask] = start[mask] + delta
            result[c] = poisoned
        return result
