"""Coordinated update attacks, applied after local training in FedHEP servers.

ALIE: mean - z * coordinate std. Min-Max: mean - gamma * p, with largest
feasible gamma such that every distance to an observed update is bounded by
its population's maximum pairwise distance (Shejwalkar & Houmansadr, NDSS 2021);
p is the inverse std (default), inverse unit vector or inverse sign direction.
IPM (inner product manipulation, Xie et al. 2020): -epsilon * mean.
Min-Sum: as Min-Max, but the attacker's summed squared distance to the observed updates may not exceed the
smallest such sum among them (official NDSS21 code). Fang's trimmed-mean attack (USENIX Security 2020), written for
update deltas: each attacker samples, per coordinate, just past the honest extreme opposite to the honest direction
(full knowledge, b = 2), or from mu -/+ [3, 4] sigma of its colluders' updates (partial knowledge).
History attacks (HistoryAttack): PoisonedFL (Xie et al., CVPR 2025) sends c |g| * s, with g the last global update
and s a random sign vector fixed for the whole run; XFED (2026, its uv variant) moves each attacker's own update by
mu against its direction, mu = median + 4 MAD of the norms of the last 8 global updates.
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
    if kind == "min_sum":
        sq = (observed ** 2).sum(1)
        n, total, sq_sum = len(observed), observed.sum(0), sq.sum()
        bound = float((n * sq - 2 * observed @ total + sq_sum).min())   # smallest summed squared distance

        def feasible(gamma: float) -> bool:
            y = mean - gamma * p
            return float(n * (y @ y) - 2 * (y @ total) + sq_sum) <= bound
    else:
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


def fang_trim(observed: torch.Tensor, full: bool, gen: torch.Generator, b: float = 2.0) -> torch.Tensor:
    """One attacker's delta for Fang's trimmed-mean attack. full: just past the honest extreme opposite to the
    honest direction (factor b); else from mu -/+ [3, 4] sigma of the colluders' own updates."""
    mean = observed.mean(0)
    up = mean > 0                                   # the honest step would raise these coordinates
    u = torch.rand(mean.shape, generator=gen).to(mean.device)
    if full:
        lo, hi = observed.min(0).values, observed.max(0).values
        below = torch.where(lo > 0, lo / b + u * (lo - lo / b), b * lo + u * (lo - b * lo))
        above = torch.where(hi > 0, hi + u * (b * hi - hi), hi + u * (hi / b - hi))
    else:
        sd = observed.std(0, correction=0)
        below, above = mean - (3 + u) * sd, mean + (3 + u) * sd
    return torch.where(up, below, above)


class CoordinatedAttack(ByzantineAttack):
    def __init__(self, kind: str, strength: float = 1.0, knowledge: str = "colluders",
                 perturbation: str = "std", seed: int = 0) -> None:
        super().__init__("label_flip")
        if (kind not in ("alie", "min_max", "ipm", "min_sum", "fang_trim")
                or knowledge not in ("colluders", "omniscient") or perturbation not in ("std", "unit", "sign")):
            raise ValueError("Unknown coordinated attack, knowledge model or perturbation")
        self.kind, self.strength, self.knowledge, self.perturbation = kind, strength, knowledge, perturbation
        self.gen = torch.Generator().manual_seed(seed)

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
        delta = None if self.kind == "fang_trim" else crafted_update(observed, self.kind, self.strength,
                                                                       self.perturbation)
        for c in sorted(attackers):
            poisoned = uploads[c].clone()
            poisoned[mask] = start[mask] + (fang_trim(observed, self.knowledge == "omniscient", self.gen)
                                            if delta is None else delta)
            result[c] = poisoned
        return result


class HistoryAttack(ByzantineAttack):
    """PoisonedFL and XFED: they need no honest updates, only the global model's history, which they read from the
    start of each round. Round 1 has no history, so the attackers upload their clean update. PoisonedFL's c stays
    at its starting value: the paper lowers it only after a failed sign-alignment test over 50 rounds, more than a
    30-round run holds; the attackers' own share of the global update is not removed from g."""

    def __init__(self, kind: str, seed: int = 0, c: float = 8.0) -> None:
        super().__init__("label_flip")
        if kind not in ("poisonedfl", "xfed"):
            raise ValueError("Unknown history attack")
        self.kind, self.c, self.seed = kind, c, seed
        self.last_start: torch.Tensor | None = None
        self.last_step: torch.Tensor | None = None
        self.norms: list[float] = []
        self.signs: torch.Tensor | None = None

    def _observe(self, start: torch.Tensor) -> None:
        """Once per round (poison_batch is called for both uploads): record the global update that led here."""
        if self.last_start is not None and torch.equal(start, self.last_start):
            return
        if self.last_start is not None:
            self.last_step = start - self.last_start
            self.norms = (self.norms + [float(self.last_step.norm())])[-8:]
        self.last_start = start.clone()

    def poison_batch(self, uploads: Mapping[int, torch.Tensor], start: torch.Tensor,
                     attackers: frozenset[int], parameter_mask: torch.Tensor) -> dict[int, torch.Tensor]:
        self._observe(start)
        result = dict(uploads)
        if not attackers or self.last_step is None:
            return result
        mask = parameter_mask.to(start.device)
        g = self.last_step[mask]
        if self.signs is None:
            gen = torch.Generator().manual_seed(self.seed)
            self.signs = (torch.randint(0, 2, g.shape, generator=gen) * 2 - 1).to(g.dtype).to(g.device)
        if self.kind == "xfed":
            norms = torch.tensor(self.norms)
            med = norms.median()
            mu = float(med + 4 * (norms - med).abs().median())
        for c in sorted(attackers):
            if self.kind == "poisonedfl":
                delta = self.c * g.abs() * self.signs
            else:
                own = (uploads[c] - start)[mask]
                delta = own - mu * own / own.norm()
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

