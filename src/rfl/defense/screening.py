"""Phase 3 screening: decide each round which clients' updates to leave out, from how far each update's direction
falls from the others', measured relative to this round's spread.

Defended FedHEP (rfl.defense.soft_cosine + trust_ledger) turns cosines into weights with a fixed temperature and
only removes a client when its momentum-averaged share stays below 0.25. Phase 2 showed why that fails: late in
training all cosines shrink together, so the weights flatten and nobody ever crosses the fixed cut. Screening
replaces the fixed cut with a robust z-score (distance from the median in units of the median absolute deviation),
which keeps its meaning as the cosines shrink:
  * score s_i = cos(signal_i, reference), on the same signal SCCF uses ("root_head": root-head rows of the classes
    the sender holds) or the whole update except BatchNorm buffers ("weights");
  * reference = mean of unit updates (as SCCF) or the coordinate-wise median update (not pulled by a minority);
  * z_i = (s_i - median s) / max(1.4826 MAD s, 0.1 |median s|); leave the client out this round if
    z_i < -z_cut. The floor stops a tight consensus (tiny MAD) from turning small differences into huge z; it
    scales with the scores, so z stays unchanged when all cosines shrink together. Fewer than 5 senders are not
    screened (a MAD of 4 values is noise);
  * evidence E_i = decay * E_i + (1 - decay) * z_i; isolate once E_i < -evidence_cut.
Phase 3b options, added after the first CIFAR-100 runs showed honest clients being isolated one after another once
the attackers were gone (each removal shrinks the population, so its spread, so the next z gets more extreme):
  * readmit: isolated clients are still scored every round, against the reference and spread of the clients
    currently kept, and come back once their evidence recovers above -evidence_cut / 2;
  * max_excluded: never leave out more than this fraction of all clients in a round (the lowest evidence go).
Phase 3c option, after the three-seed runs showed readmission brings attackers back late in training (when every
update's direction carries little information) and does not stop the cascade (removed honest clients are judged
against the same shrunken group):
  * population = "all": the reference and the median/MAD are taken over every client not banned, isolated ones
    included, so removing clients never shrinks the yardstick. Median and MAD tolerate a minority of attackers in
    that population; isolation stays permanent."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from rfl.defense.soft_cosine import FlatLayout, bound_update_norms, coordinate_median


@dataclass(frozen=True)
class ScreenOptions:
    signal: str = "root_head"           # "root_head" | "weights"
    reference: str = "mean_direction"   # "mean_direction" | "median"
    z_cut: float = 2.5
    evidence_cut: float = 1.5
    evidence_decay: float = 0.7
    norm_bound: float = 3.0             # cap at k x the lower-quartile norm before scoring, as SCCF does
    readmit: bool = False
    max_excluded: float = 1.0
    population: str = "kept"            # "kept" (clients not isolated) | "all" (every client not banned)
    drill_down: bool = False            # Phase 4: check parents, then their members more strictly
    group_cut: float = 2.0              # flag a parent when its mean update's z (vs random groups) < -group_cut
    drill_cut: float = 1.0              # members of a flagged parent are left out this round when z < -drill_cut
    drill_mode: str = "exclude"         # "exclude" (Phase 4 v1) | "evidence" (4b: member alignment feeds evidence)
    min_group: int = 2                  # parents with fewer eligible members are not checked
    mad_floor: float = 0.0             # absolute cosine-score scale floor; 0 preserves previous runs
    relative_floor: float = 0.1
    isolation_patience: int = 1        # consecutive z < -evidence_cut, also requiring EMA below cut
    exclusion_patience: int = 1        # consecutive z < -z_cut before temporary exclusion


    def __post_init__(self) -> None:
        if self.mad_floor < 0 or self.relative_floor < 0:
            raise ValueError("MAD floors must be nonnegative")
        if self.isolation_patience < 1 or self.exclusion_patience < 1:
            raise ValueError("Patience must be positive")
        if self.signal not in ("root_head", "weights"):
            raise ValueError(f"Unknown screening signal {self.signal!r}")
        if self.reference not in ("mean_direction", "median"):
            raise ValueError(f"Unknown screening reference {self.reference!r}")
        if self.population not in ("kept", "all"):
            raise ValueError(f"Unknown screening population {self.population!r}")
        if self.drill_mode not in ("exclude", "evidence"):
            raise ValueError(f"Unknown drill-down mode {self.drill_mode!r}")


def _signal(updates: Sequence[torch.Tensor], layout: FlatLayout, class_masks: torch.Tensor | None,
            opts: ScreenOptions) -> tuple[torch.Tensor, torch.Tensor | None]:
    """(signal vectors, their class masks or None)."""
    stack = torch.stack(list(updates)).float()
    params = bound_update_norms(stack[:, ~layout.buffer_mask.to(stack.device)], opts.norm_bound)
    vecs, mask = params, None
    if opts.signal == "root_head" and class_masks is not None and layout.classifier is not None:
        s, e = layout.classifier
        C = class_masks.shape[1]
        n = params.shape[0]
        mask = class_masks.float().to(params.device).unsqueeze(2).expand(n, C, (e - s) // C).reshape(n, -1)
        vecs = params[:, s:e] * mask
    return vecs, mask


def _reference(vecs: torch.Tensor, opts: ScreenOptions, reference_rows: torch.Tensor | None) -> torch.Tensor:
    base = vecs if reference_rows is None else vecs[reference_rows.to(vecs.device)]
    if opts.reference == "median":
        return coordinate_median(base)
    return F.normalize(base, dim=1, eps=1e-10).mean(dim=0)


def screening_scores(updates: Sequence[torch.Tensor], layout: FlatLayout, class_masks: torch.Tensor | None,
                     opts: ScreenOptions, reference_rows: torch.Tensor | None = None) -> torch.Tensor:
    """Cosine of each sender's (norm-bounded) update to the reference, on the chosen signal. The reference is
    built from the rows in `reference_rows` (bool), default all."""
    vecs, mask = _signal(updates, layout, class_masks, opts)
    ref = _reference(vecs, opts, reference_rows)
    target = ref.unsqueeze(0) * mask if mask is not None else ref.unsqueeze(0).expand_as(vecs)
    return F.cosine_similarity(vecs, target, dim=1, eps=1e-10)


def group_z(updates: Sequence[torch.Tensor], layout: FlatLayout, class_masks: torch.Tensor | None,
            opts: ScreenOptions, groups: Sequence[Sequence[int]], reference_rows: torch.Tensor | None = None,
            generator: torch.Generator | None = None, draws: int = 64) -> torch.Tensor:
    """Phase 4: how out of line each group's mean update is. Score = cosine of the group's mean update (rows of
    `updates` listed per group) to the clients' reference. Averaging adds up what members share, so attackers
    pushing one direction (e.g. one backdoor) stand out more together than one by one. Averaging also makes any
    mean more aligned than a single client, so each group is compared with random groups of the same size drawn
    from the reference rows (a permutation null):
    z = (score - median of null) / max(1.4826 MAD of null, 0.01 |median of null|). (Means of random groups vary
    far less than single clients, so the floor is 10x lower than for clients.)
    `generator` keeps the draws off the global RNG (training randomness stays untouched)."""
    vecs, mask = _signal(updates, layout, class_masks, opts)
    ref = _reference(vecs, opts, reference_rows)
    pool = torch.arange(vecs.shape[0]) if reference_rows is None else torch.nonzero(reference_rows).flatten()

    def score(idx: torch.Tensor) -> torch.Tensor:
        idx = idx.to(vecs.device)
        target = ref * mask[idx].amax(dim=0) if mask is not None else ref
        return F.cosine_similarity(vecs[idx].mean(dim=0), target, dim=0, eps=1e-10)

    out = []
    for rows in groups:
        m = len(rows)
        if m < 2 or len(pool) <= m:
            out.append(torch.tensor(0.0))
            continue
        null = torch.stack([score(pool[torch.randperm(len(pool), generator=generator)[:m]]) for _ in range(draws)])
        med = null.median()
        scale = torch.maximum((null - med).abs().median() * 1.4826, 0.01 * med.abs())
        s = score(torch.tensor(list(rows), dtype=torch.long))
        out.append((s - med) / scale if scale > 1e-12 else torch.tensor(0.0))
    return torch.stack(out).cpu() if out else torch.zeros(0)


def member_alignment(updates: Sequence[torch.Tensor], layout: FlatLayout, class_masks: torch.Tensor | None,
                     opts: ScreenOptions, members: Sequence[int], outside: Sequence[int],
                     reference_rows: torch.Tensor | None = None) -> torch.Tensor:
    """Phase 4b: which members of a flagged parent share its unusual direction. Residual r_i = signal_i minus the
    reference. Member i's score is cos(r_i, mean of the other members' residuals); attackers sharing one direction
    score high, an honest member next to them does not. Null: the same cosine for clients outside the parent,
    against each of the leave-one-out directions (same averaging size). Returns the members' robust z against
    that null (rows of `updates` given by `members` / `outside`)."""
    vecs, mask = _signal(updates, layout, class_masks, opts)
    ref = _reference(vecs, opts, reference_rows)
    res = vecs - (ref.unsqueeze(0) * mask if mask is not None else ref.unsqueeze(0))
    rows = torch.tensor(list(members), dtype=torch.long, device=vecs.device)
    out_rows = torch.tensor(list(outside), dtype=torch.long, device=vecs.device)
    total = res[rows].sum(dim=0)
    own, null = [], []
    for i in range(len(rows)):
        d = (total - res[rows[i]]) / (len(rows) - 1)
        own.append(F.cosine_similarity(res[rows[i]], d, dim=0, eps=1e-10))
        if len(out_rows):
            null.append(F.cosine_similarity(res[out_rows], d.unsqueeze(0), dim=1, eps=1e-10))
    if not null:
        return torch.zeros(len(rows))
    return robust_z(torch.stack(own).cpu(), reference=torch.cat(null).cpu())


MIN_SENDERS = 5


def robust_z(scores: torch.Tensor, floor: float = 0.1, reference: torch.Tensor | None = None,
             absolute_floor: float = 0.0) -> torch.Tensor:
    """(s - median) / max(1.4826 MAD, floor * |median|), median and MAD taken over `reference` (default:
    `scores` itself); all zeros with fewer than MIN_SENDERS reference scores or no spread."""
    ref = scores if reference is None else reference
    if ref.numel() < MIN_SENDERS:
        return torch.zeros_like(scores)
    med = ref.median()
    scale = torch.maximum((ref - med).abs().median() * 1.4826, floor * med.abs()).clamp_min(absolute_floor)
    no_spread = scale <= 1e-9 * ref.abs().max()  # relative, so the cutoff itself does not depend on scale
    return torch.zeros_like(scores) if no_spread else (scores - med) / scale


class EvidenceTracker:
    def __init__(self, decay: float, cut: float, readmit: bool = False, patience: int = 1) -> None:
        self.decay, self.cut, self.readmit = decay, cut, readmit
        self.patience = patience
        self.streak: dict[int, int] = {}
        self.evidence: dict[int, float] = {}
        self.isolated: set[int] = set()

    def update(self, z: Mapping[int, float]) -> None:
        for c, v in z.items():
            self.evidence[c] = self.decay * self.evidence.get(c, 0.0) + (1.0 - self.decay) * v
            self.streak[c] = self.streak.get(c, 0) + 1 if v < -self.cut else 0
            if self.evidence[c] < -self.cut and (self.patience == 1 or self.streak[c] >= self.patience):
                self.isolated.add(c)
            elif self.readmit and c in self.isolated and self.evidence[c] > -self.cut / 2:
                self.isolated.discard(c)


def apply_cap(excluded: set[int], evidence: Mapping[int, float], z: Mapping[int, float], limit: int) -> set[int]:
    """At most `limit` clients stay excluded: those with the lowest evidence (then lowest z)."""
    if len(excluded) <= limit:
        return excluded
    order = sorted(excluded, key=lambda c: (evidence.get(c, 0.0), z.get(c, 0.0)))
    return set(order[:limit])
