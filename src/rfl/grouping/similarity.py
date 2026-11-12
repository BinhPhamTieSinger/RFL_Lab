"""Client -> parent assignment by update similarity, ported from the authors' `HierarchicalTopology`
(src/topologies/hierarchical.py) and the clustering controller in `HierarchicalEnsembleEngine.run_round`.

Clients start in random groups. Once their update directions settle (or after `warmup_max` rounds), they are
regrouped once by k-means on randomly projected update vectors; afterwards they are regrouped whenever too many
clients point closer to another group's mean update than to their own. Groups are numbered 0..K-1 (the authors'
head ids -2-k)."""
from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from enum import Enum

import numpy as np
import torch


def random_assignment(num_clients: int, num_groups: int, seed: int) -> dict[int, int]:
    """Authors' `HierarchicalTopology.build` without label information."""
    rng = np.random.RandomState(seed)
    return {i: int(rng.randint(0, num_groups)) for i in range(num_clients)}


def concentrate(assignment: Mapping[int, int], chosen: Collection[int], num_groups: int, group: int = 0
                ) -> dict[int, int]:
    """Moves every client in `chosen` into `group` and keeps that group at most ceil(N / K) strong by moving its
    other members, in id order, to the currently smallest other groups. Everyone else keeps their group. Used to
    place all attackers in one parent."""
    out = dict(assignment)
    cap = max(len(chosen), -(-len(out) // num_groups))
    for c in chosen:
        out[c] = group
    for c in sorted(out):
        if out[c] == group and c not in chosen and sum(g == group for g in out.values()) > cap:
            sizes = {g: sum(v == g for v in out.values()) for g in range(num_groups) if g != group}
            out[c] = min(sizes, key=lambda g: (sizes[g], g))
    return out


def spread(assignment: Mapping[int, int], chosen: Collection[int], num_groups: int) -> dict[int, int]:
    """Deals the clients in `chosen` round-robin over the groups (in id order), so no group holds more than
    ceil(len(chosen) / K) of them. Everyone else keeps their group."""
    out = dict(assignment)
    for i, c in enumerate(sorted(chosen)):
        out[c] = i % num_groups
    return out


def random_project(vectors: np.ndarray, target_dim: int = 256, seed: int = 42, chunk_rows: int = 65536
                   ) -> np.ndarray:
    """vectors @ R with R ~ N(0, 1/target_dim), shape (D, target_dim), drawn from RandomState(seed). Same matrix as
    the authors' `_random_project`, but generated `chunk_rows` rows at a time: theirs allocates all of R at once
    (3.5 GB for ResNet-9). RandomState draws normals in a fixed stream, so the chunks are exactly their rows."""
    n, d = vectors.shape
    if d <= target_dim:
        return vectors
    rng = np.random.RandomState(seed)
    out = np.zeros((n, target_dim), dtype=np.float64)
    for start in range(0, d, chunk_rows):
        stop = min(d, start + chunk_rows)
        out += vectors[:, start:stop] @ (rng.randn(stop - start, target_dim) / np.sqrt(target_dim))
    return out


def kmeans_on_updates(deltas: Mapping[int, torch.Tensor], num_groups: int, seed: int) -> dict[int, int]:
    """Authors' `build_update_similarity`: project, L2-normalise (so Euclidean k-means groups by direction),
    initialise centroids at `num_groups` distinct clients, run 10 iterations."""
    ids = sorted(deltas)
    matrix = np.stack([deltas[c].detach().cpu().numpy() for c in ids])
    projected = random_project(matrix, 256, seed)
    vectors = projected / np.maximum(np.linalg.norm(projected, axis=1, keepdims=True), 1e-8)
    k = min(num_groups, len(ids))
    rng = np.random.RandomState(seed)
    centroids = vectors[rng.choice(len(ids), k, replace=False)].copy()
    assign = np.zeros(len(ids), dtype=int)
    for _ in range(10):
        assign = np.argmin(np.linalg.norm(vectors[:, None, :] - centroids[None, :, :], axis=2), axis=1)
        for g in range(k):
            members = vectors[assign == g]
            if len(members) > 0:
                centroids[g] = members.mean(axis=0)
    return {c: int(a) for c, a in zip(ids, assign, strict=True)}


def _cos(a: torch.Tensor, b: torch.Tensor) -> float | None:
    na, nb = torch.norm(a), torch.norm(b)
    if na <= 1e-8 or nb <= 1e-8:
        return None
    return (torch.dot(a, b) / (na * nb)).item()


def group_centroids(assignment: Mapping[int, int], deltas: Mapping[int, torch.Tensor], num_groups: int
                    ) -> dict[int, torch.Tensor]:
    """Mean update of each non-empty group."""
    out = {}
    for g in range(num_groups):
        members = [deltas[c] for c in sorted(deltas) if assignment[c] == g]
        if members:
            out[g] = torch.stack(members).mean(dim=0)
    return out


def mean_member_similarity(assignment: Mapping[int, int], deltas: Mapping[int, torch.Tensor], num_groups: int
                           ) -> float:
    """Mean cosine of each member's update to its group's mean, over groups with 2+ members. Authors'
    `_avg_intra_cluster_similarity`; above 0.5 their parents copy the global model instead of averaging."""
    sims: list[float] = []
    for g in range(num_groups):
        members = [c for c in sorted(deltas) if assignment[c] == g]
        if len(members) < 2:
            continue
        centroid = torch.stack([deltas[c] for c in members]).mean(dim=0)
        sims.extend(s for c in members if (s := _cos(deltas[c], centroid)) is not None)
    return float(np.mean(sims)) if sims else 0.0


def count_misaligned(assignment: Mapping[int, int], deltas: Mapping[int, torch.Tensor], num_groups: int,
                     margin: float) -> int:
    """Clients whose update is closer (cosine) to another group's mean than to their own by more than `margin`.
    Authors' `_count_misaligned_clients`."""
    centroids = group_centroids(assignment, deltas, num_groups)
    if len(centroids) <= 1:
        return 0
    count = 0
    for c in sorted(deltas):
        sims = {g: s for g, cent in centroids.items() if (s := _cos(deltas[c], cent)) is not None}
        own = assignment[c]
        if own not in sims:
            continue
        best_other = max((s for g, s in sims.items() if g != own), default=-1.0)
        count += best_other - sims[own] > margin
    return count


class Regroup(Enum):
    NONE = "none"
    INITIAL = "initial"  # first similarity grouping: the server resets parents and drops clients' local heads
    UPDATE = "update"    # a later regrouping: parents keep their weights


class SimilarityGrouping:
    """Holds the current assignment and decides each round whether to regroup."""

    def __init__(self, num_clients: int, num_groups: int, seed: int, warmup_min: int = 2, warmup_max: int = 10,
                 stability: float = 0.30, misalignment: float = 0.15, misaligned_ratio: float = 0.15) -> None:
        self.num_groups = num_groups
        self.seed = seed
        self.warmup_min, self.warmup_max = warmup_min, warmup_max
        self.stability, self.misalignment, self.misaligned_ratio = stability, misalignment, misaligned_ratio
        self.assignment = random_assignment(num_clients, num_groups, seed)
        self.has_grouped = False
        self._previous: dict[int, torch.Tensor] = {}

    def members(self, group: int, among: Sequence[int]) -> list[int]:
        return [c for c in among if self.assignment[c] == group]

    def step(self, deltas: Mapping[int, torch.Tensor], round_idx: int) -> Regroup:
        """`deltas`: each client's flat update this round (see rfl.core.state). Regroups in place if due."""
        event = Regroup.NONE
        if not self.has_grouped:
            if round_idx >= self.warmup_min:
                sims = [s for c in deltas if c in self._previous
                        if (s := _cos(self._previous[c], deltas[c])) is not None]
                stability = float(np.mean(sims)) if sims else 0.0
                if stability >= self.stability or round_idx >= self.warmup_max:
                    event = Regroup.INITIAL
        elif count_misaligned(self.assignment, deltas, self.num_groups, self.misalignment) / len(deltas) \
                > self.misaligned_ratio:
            event = Regroup.UPDATE
        if event is not Regroup.NONE:
            self.assignment.update(kmeans_on_updates(deltas, self.num_groups, self.seed + round_idx))
            self.has_grouped = True
        self._previous.update(deltas)
        return event
