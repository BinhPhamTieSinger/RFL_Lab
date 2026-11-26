"""Coordinated update attacks, applied after local training in FedHEP servers.

ALIE: mean - z * coordinate std. Min-Max: mean - gamma * p, with largest
feasible gamma such that every distance to an observed update is bounded by
its population's maximum pairwise distance (Shejwalkar & Houmansadr, NDSS 2021);
p is the inverse std (default), inverse unit vector or inverse sign direction.
IPM (inner product manipulation, Xie et al. 2020): -epsilon * mean.
Statistics use colluders' clean updates by default; omniscient mode uses honest
updates. Neither attack promises evasion of this defense's cosine/EMA threshold.
BatchNorm buffers are preserved from each attacker's clean upload.

Stealth backdoor (adaptive, omniscient): the attacker trains on stamped images as
in BadNets, then pulls its update toward the honest mean, m + lambda (u - m), with
the largest lambda in [0, 1] that keeps it within Min-Max's bound (no farther from
any honest update than the farthest honest pair). Distance-bounded poisoning in the
spirit of constrain-and-scale (Bagdasaryan et al., 2020) and Min-Max; it does not
read the screen's own statistic.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch

from rfl.attacks.backdoor import BackdoorAttack
from rfl.attacks.byzantine import ByzantineAttack


def _within_bound(observed: torch.Tensor) -> float:
    """Min-Max's radius: the largest squared distance between two observed updates."""
    return max(float(((row - observed) ** 2).sum(1).max()) for row in observed)


def crafted_update(observed: torch.Tensor, kind: str, strength: float = 1.0, perturbation: str = "std"
                   ) -> torch.Tensor:
    mean = observed.mean(0)
    if kind == "ipm":
        return -strength * mean
    std = observed.std(0, correction=0)
    if kind == "alie":
        return mean - strength * std
    p = {"std": std, "unit": mean / mean.norm(), "sign": torch.sign(mean)}[perturbation]
    # squared distances avoid allocating an n x n x dimension tensor
    radius = _within_bound(observed)

    def feasible(gamma: float) -> bool:
        return float(((mean - gamma * p - observed) ** 2).sum(1).max()) <= radius

    lo, hi = 0.0, 1.0
    while feasible(hi) and hi < 1e6:
        lo, hi = hi, hi * 2
    for _ in range(40):
        mid = (lo + hi) / 2
        if feasible(mid):
            lo = mid
        else:
            hi = mid
    return mean - lo * p


class CoordinatedAttack(ByzantineAttack):
    def __init__(self, kind: str, strength: float = 1.0, knowledge: str = "colluders",
                 perturbation: str = "std") -> None:
        super().__init__("label_flip")
        if (kind not in ("alie", "min_max", "ipm") or knowledge not in ("colluders", "omniscient")
                or perturbation not in ("std", "unit", "sign")):
            raise ValueError("Unknown coordinated attack, knowledge model or perturbation")
        self.kind, self.strength, self.knowledge, self.perturbation = kind, strength, knowledge, perturbation

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
        delta = crafted_update(observed, self.kind, self.strength, self.perturbation)
        for c in attackers:
            poisoned = uploads[c].clone()
            poisoned[mask] = start[mask] + delta
            result[c] = poisoned
        return result


def stealth_blend(update: torch.Tensor, honest: torch.Tensor) -> tuple[torch.Tensor, float]:
    """m + lambda (update - m) with the largest lambda in [0, 1] inside Min-Max's bound; returns it and lambda."""
    mean = honest.mean(0)
    radius = _within_bound(honest)
    direction = update - mean

    def feasible(lam: float) -> bool:
        return float(((mean + lam * direction - honest) ** 2).sum(1).max()) <= radius

    if feasible(1.0):
        return update, 1.0
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if feasible(mid) else (lo, mid)
    return mean + lo * direction, lo


class StealthBackdoorAttack(BackdoorAttack):
    """BadNets training, then each attacker's update is blended toward the honest mean (stealth_blend)."""

    def __init__(self, target: int = 0, fraction: float = 0.5, seed: int = 0) -> None:
        super().__init__(target, fraction, 1.0, seed)
        self.blend: list[float] = []   # lambda of every attacker upload, for inspection

    def poison_batch(self, uploads: Mapping[int, torch.Tensor], start: torch.Tensor,
                     attackers: frozenset[int], parameter_mask: torch.Tensor) -> dict[int, torch.Tensor]:
        result = dict(uploads)
        honest = sorted(set(uploads) - attackers)
        if not attackers or len(honest) < 2:
            return result
        mask = parameter_mask.to(start.device)
        observed = torch.stack([(uploads[c] - start)[mask] for c in honest])
        for c in sorted(attackers & set(uploads)):
            delta, lam = stealth_blend((uploads[c] - start)[mask], observed)
            self.blend.append(lam)
            poisoned = uploads[c].clone()
            poisoned[mask] = start[mask] + delta
            result[c] = poisoned
        return result

