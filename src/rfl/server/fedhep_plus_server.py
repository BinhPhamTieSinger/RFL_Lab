"""FedHEP+ (Phase 3): Defended FedHEP with the failure points found in Phase 2 replaced, each switchable so every
change can be tested on its own against the faithful Defended FedHEP.

Per round:
 1. every client trains (as in FedHEP);
 2. screening (rfl.defense.screening) scores every non-isolated client's update against this round's reference,
    leaves out clients far below the others (robust z < -z_cut) and isolates for good those that stay low;
 3. optionally, clients are regrouped by the similarity of their weight updates (instead of staying in their random
    initial groups), using the kept clients only;
 4. parents and Global aggregate the kept clients with SCCF exactly as Defended FedHEP does (same weights, norm
    bounds and hard rejection). Its trust ledger is not used.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from rfl.client.fedhep import FedHEPTrainer
from rfl.core.interfaces import Attack
from rfl.core.types import ClientData, DetectionResult, RoundReport
from rfl.defense.classic import arc_nnm_trimmed_mean_rule
from rfl.defense.probe import probe_outputs
from rfl.defense.screening import (
    EvidenceTracker,
    ScreenOptions,
    apply_cap,
    gap_ratio,
    group_z,
    member_alignment,
    one_sided,
    robust_z,
    screening_scores,
)
from rfl.defense.soft_cosine import coordinate_median
from rfl.grouping.similarity import Regroup, SimilarityGrouping
from rfl.models.resnet9 import MultiHeadResNet9
from rfl.server.defended_fedhep_server import DefendedFedHEPServer


class FedHEPPlusServer(DefendedFedHEPServer):
    def __init__(self, model: MultiHeadResNet9, clients: Sequence[ClientData], trainer: FedHEPTrainer,
                 grouping: SimilarityGrouping, screen: ScreenOptions, regroup: bool = False,
                 attack: Attack | None = None, attackers: frozenset[int] = frozenset(),
                 global_rule: str = "sccf", global_f: int = 3, probe: torch.Tensor | None = None) -> None:
        super().__init__(model, clients, trainer, grouping, attack=attack, attackers=attackers)
        self.screen = screen
        self.regroup = regroup
        if global_rule not in ("sccf", "arc_nnm_tm"):
            raise ValueError(f"unknown global_rule {global_rule!r}")
        self.global_rule, self.global_f = global_rule, global_f
        self.probe = probe   # images only; required for the output-space signals
        if screen.signal in ("probe_logits", "head_gap") and probe is None:
            raise ValueError(f"signal {screen.signal!r} needs a probe set")
        self.similarity_mask = ~self.global_defense.layout.buffer_mask  # regrouping compares weights only
        self.evidence = EvidenceTracker(screen.evidence_decay, screen.evidence_cut, screen.readmit,
                                        screen.isolation_patience)
        self.exclusion_streak: dict[int, int] = {}
        self.banned: set[int] = set()  # non-finite uploads: out for good, even with readmission

    def run_round(self, round_idx: int) -> None:
        start = self.global_vec.clone()
        ids = list(range(len(self.clients)))
        to_global, to_parent = self._train_all(start, round_idx)

        isolated = self.evidence.isolated
        for c in ids:  # sentinel, as in Defended FedHEP
            if not (torch.isfinite(to_global[c]).all() and torch.isfinite(to_parent[c]).all()):
                self.banned.add(c)
                isolated.add(c)
        # Without readmission isolated clients are no longer scored; with it, everyone not banned is scored
        # against the reference and spread of the clients currently kept.
        everyone = self.screen.readmit or self.screen.population == "all"
        scored = [c for c in ids if c not in self.banned and (everyone or c not in isolated)]

        score_stats: dict[str, object] = {}
        raw_scores: dict[int, float] = {}
        z: dict[int, float] = {}
        z_logged: dict[int, float] = {}
        parent_z: dict[int, float] = {}
        flagged_groups: set[int] = set()
        align_z: dict[int, float] = {}
        if len(scored) > 1:
            in_ref = torch.tensor([self.screen.population == "all" or c not in isolated for c in scored])
            updates = [to_global[c] - start for c in scored]
            if self.screen.signal in ("probe_logits", "head_gap"):
                scores = self._probe_scores(to_global, scored, in_ref)
            else:
                scores = screening_scores(updates, self.global_defense.layout, self._masks(scored), self.screen,
                                          reference_rows=in_ref)
            ref_scores = scores[in_ref]
            if ref_scores.numel():
                med = ref_scores.median()
                mad = (ref_scores - med).abs().median()
                scale = max(1.4826 * float(mad), self.screen.relative_floor * abs(float(med)),
                            self.screen.mad_floor)
                score_stats = {"median": float(med), "mad": float(mad), "scale": scale,
                               "relative_floor": self.screen.relative_floor * abs(float(med)),
                               "absolute_floor": self.screen.mad_floor, "reference_n": int(in_ref.sum())}
            raw_scores = dict(zip(scored, scores.tolist(), strict=True))
            zs = robust_z(scores, floor=self.screen.relative_floor, reference=ref_scores,
                          absolute_floor=self.screen.mad_floor)
            z_logged = {c: float(v) for c, v in zip(scored, zs.tolist(), strict=True)}
            z = one_sided(z_logged, self.screen.two_sided)
            split = gap_ratio(scores.tolist())
            score_stats["gap_ratio"] = split
            if self.screen.gap_gate > 0 and split < self.screen.gap_gate:
                z = {c: max(v, 0.0) for c, v in z.items()}   # no separate group this round: no bad evidence
            z_evidence = dict(z)
            if self.screen.drill_down:
                parent_z = self._group_z(updates, scored, isolated, in_ref, round_idx)
                flagged_groups = {g for g, v in parent_z.items() if v < -self.screen.group_cut}
                if self.screen.drill_mode == "evidence":
                    # 4b: members sharing their flagged parent's unusual direction get that as evidence.
                    align_z = self._align(updates, scored, isolated, in_ref, flagged_groups)
                    for c, a in align_z.items():
                        z_evidence[c] = min(z[c], -a)
            self.evidence.update(z_evidence)
        for c, v in z.items():
            self.exclusion_streak[c] = self.exclusion_streak.get(c, 0) + 1 if v < -self.screen.z_cut else 0
        left_out = {c for c, v in z.items() if v < -self.screen.z_cut
                    and self.exclusion_streak[c] >= self.screen.exclusion_patience}
        if self.screen.drill_mode == "exclude":
            # v1 drill-down: members of a flagged parent face a stricter cut this round.
            left_out |= {c for c, v in z.items()
                         if self.grouping.assignment[c] in flagged_groups and v < -self.screen.drill_cut}
        limit = int(self.screen.max_excluded * len(ids))
        excluded = self.banned | apply_cap((left_out | isolated) - self.banned, self.evidence.evidence, z, limit)
        kept = [c for c in ids if c not in excluded]

        event = Regroup.NONE
        if self.regroup and kept:
            event = self.grouping.step(self._for_similarity({c: to_global[c] - start for c in kept}), round_idx)
            if event is Regroup.INITIAL:
                self.parents = {g: start.clone() for g in self.parents}
                for c in ids:
                    self.memory[c].local_head = None

        for g in self.parents:
            members = self.grouping.members(g, kept)
            if members:
                ref = self.parents[g]
                updates = [to_parent[c] - ref for c in members]
                step, _ = self.parent_defense.aggregate(updates, self._masks(members))
                self.parents[g] = ref + step

        global_trust: dict[int, float] = {}
        if kept and self.global_rule == "arc_nnm_tm":
            # Layer L0 under the screen: the attackers still assumed present are those not yet excluded.
            f = max(0, min(self.global_f - len(excluded), (len(kept) - 1) // 2))
            step = arc_nnm_trimmed_mean_rule(torch.stack([to_global[c] - start for c in kept]), f).update
            self.global_vec = start + step
            global_trust = {c: 1.0 / len(kept) for c in kept}
        elif kept:
            step, weights = self.global_defense.aggregate([to_global[c] - start for c in kept], self._masks(kept))
            self.global_vec = start + step
            global_trust = {c: float(t) for c, t in zip(kept, weights.tolist(), strict=True)}

        trust = {**{c: -99.0 for c in isolated}, **z}  # isolated and not scored: ranked below everyone
        self.report = RoundReport(DetectionResult(trust, frozenset(excluded)),
                                  groups=dict(self.grouping.assignment),
                                  flagged_groups=frozenset(flagged_groups) if self.screen.drill_down else None)
        self.diagnostics.append({"round": round_idx, "attackers": sorted(self.attackers), "z": z_logged,
                                 "scores": raw_scores, "score_stats": score_stats,
                                 "left_out": sorted(left_out), "isolated": sorted(isolated),
                                 "excluded": sorted(excluded), "group_z": parent_z,
                                 "flagged_groups": sorted(flagged_groups), "align_z": align_z,
                                 "evidence": dict(self.evidence.evidence), "global_trust": global_trust,
                                 "regroup": event.value, "groups": dict(self.grouping.assignment)})

    def _probe_scores(self, uploads: dict[int, torch.Tensor], scored: list[int], in_ref: torch.Tensor
                      ) -> torch.Tensor:
        """Output-space scores on the uploaded models (what the server has), see rfl.defense.probe. probe_logits:
        cosine of each upload's probe logits to the coordinate median of the reference rows' logits. head_gap: the
        scalar gap itself (the screen's robust z handles scale)."""
        assert self.probe is not None
        outs = [probe_outputs(self.model, self.template, uploads[c], self.probe) for c in scored]
        if self.screen.signal == "head_gap":
            return torch.tensor([g for _, g in outs])
        logits = torch.stack([v for v, _ in outs])
        ref = coordinate_median(logits[in_ref.to(logits.device)])
        return F.cosine_similarity(logits, ref.unsqueeze(0), dim=1, eps=1e-10).cpu()

    def _group_z(self, updates: list[torch.Tensor], scored: list[int], isolated: set[int], in_ref: torch.Tensor,
                 round_idx: int) -> dict[int, float]:
        """Each parent's mean update (over its scored, not isolated members) against random groups of the same
        size (rfl.defense.screening.group_z)."""
        rows = {g: [i for i, c in enumerate(scored) if self.grouping.assignment[c] == g and c not in isolated]
                for g in self.parents}
        rows = {g: r for g, r in rows.items() if len(r) >= max(2, self.screen.min_group)}
        if not rows:
            return {}
        gen = torch.Generator().manual_seed(round_idx)
        gz = group_z(updates, self.global_defense.layout, self._masks(scored), self.screen, list(rows.values()),
                     reference_rows=in_ref, generator=gen)
        return {g: float(v) for g, v in zip(rows, gz.tolist(), strict=True)}

    def _align(self, updates: list[torch.Tensor], scored: list[int], isolated: set[int], in_ref: torch.Tensor,
               flagged: set[int]) -> dict[int, float]:
        """Members of each flagged parent: alignment z with the other members' shared deviation, against clients
        outside the parent (rfl.defense.screening.member_alignment)."""
        out: dict[int, float] = {}
        active = [i for i, c in enumerate(scored) if c not in isolated]
        for g in sorted(flagged):
            members = [i for i in active if self.grouping.assignment[scored[i]] == g]
            outside = [i for i in active if self.grouping.assignment[scored[i]] != g]
            za = member_alignment(updates, self.global_defense.layout, self._masks(scored), self.screen, members,
                                  outside, reference_rows=in_ref)
            out.update({scored[i]: float(v) for i, v in zip(members, za.tolist(), strict=True)})
        return out
