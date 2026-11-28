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

import math
from collections.abc import Sequence
from dataclasses import replace

import torch
import torch.nn.functional as F

from rfl.client.fedhep import FedHEPTrainer
from rfl.core.interfaces import Attack
from rfl.core.types import ClientData, DetectionResult, RoundReport
from rfl.defense.classic import arc_nnm_trimmed_mean_rule
from rfl.defense.probe import probe_outputs
from rfl.defense.screening import (
    PERSONAL_HEADS,
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

AGREE_ROUNDS = 5   # two-signal: both signals below -evidence_cut in one of a client's last 5 scored rounds (8.2)


class FedHEPPlusServer(DefendedFedHEPServer):
    def __init__(self, model: MultiHeadResNet9, clients: Sequence[ClientData], trainer: FedHEPTrainer,
                 grouping: SimilarityGrouping, screen: ScreenOptions, regroup: bool = False,
                 attack: Attack | None = None, attackers: frozenset[int] = frozenset(),
                 global_rule: str = "sccf", global_f: int = 3, probe: torch.Tensor | None = None,
                 global_f_fixed: bool = False, rewind: str = "none", rewind_window: int = 10,
                 rewind_agree: bool = False, global_f_floor: float = 0.0) -> None:
        super().__init__(model, clients, trainer, grouping, attack=attack, attackers=attackers)
        if rewind not in ("none", "trunk", "trunk_root", "all"):
            raise ValueError(f"unknown rewind {rewind!r}")
        self.global_f_fixed = global_f_fixed
        self.global_f_floor = global_f_floor   # share of global_f kept as the least f however many are excluded
        self.rewind_agree = rewind_agree       # rewind only for clients flagged by both signals (two-signal)
        self.rewind, self.rewind_window = rewind, rewind_window
        self.rewind_mask = self._rewind_mask(rewind)   # which coordinates a rewind restores
        self.stored: dict[int, torch.Tensor] = {}      # round -> Global model at the start of that round
        self.rewound: list[dict[str, int]] = []        # (round, target) of every rewind, for the diagnostics
        self.screen = screen
        self.regroup = regroup
        if global_rule not in ("sccf", "arc_nnm_tm"):
            raise ValueError(f"unknown global_rule {global_rule!r}")
        self.global_rule, self.global_f = global_rule, global_f
        self.probe = probe   # images only; required for the output-space signals
        if screen.signal in ("probe_logits", "head_gap") and probe is None:
            raise ValueError(f"signal {screen.signal!r} needs a probe set")
        self.similarity_mask = ~self.global_defense.layout.buffer_mask  # regrouping compares weights only
        if screen.sentence and rewind != "none":
            raise ValueError("sentence (readmission) is not combined with the rewind: a rewind after a second "
                             "isolation would reach back past the client's return")
        union = screen.signal in ("weights_critical", "shared_critical")
        if rewind_agree and (rewind == "none" or not union):
            raise ValueError("rewind_agree needs a rewind and a union signal (weights_critical, shared_critical)")
        if screen.sentence_single and not union:
            raise ValueError("sentence_single needs a union signal (weights_critical, shared_critical)")
        if not 0.0 <= global_f_floor <= 1.0:
            raise ValueError("global_f_floor is a share of global_f, in [0, 1]")
        self.evidence = EvidenceTracker(screen.evidence_decay, screen.evidence_cut, screen.readmit,
                                        screen.isolation_patience, screen.sentence, screen.max_strikes,
                                        screen.probation_patience, screen.isolation_window)
        # union signals: per client, whether both signals' z (after the gate) fell below -evidence_cut, for its
        # last AGREE_ROUNDS scored rounds; one such round makes it "two-signal" (rewind_agree, sentence_single)
        self.both_recent: dict[int, list[bool]] = {}
        self.exclusion_streak: dict[int, int] = {}
        self.banned: set[int] = set()  # non-finite uploads: out for good, even with readmission

    def two_signal(self) -> set[int]:
        """Clients flagged by both signals in at least one of their last AGREE_ROUNDS scored rounds."""
        return {c for c, v in self.both_recent.items() if any(v)}

    def _rewind_mask(self, rewind: str) -> torch.Tensor | None:
        """Coordinates a rewind restores: the trunk (no head), the trunk and the shared root head, or all."""
        if rewind == "none":
            return None
        heads: tuple[str, ...] = ("fc2_root",) + PERSONAL_HEADS
        exclude = {"trunk": heads, "trunk_root": PERSONAL_HEADS, "all": ()}[rewind]
        keep = torch.ones(self.global_vec.numel(), dtype=torch.bool, device=self.global_vec.device)
        offset = 0
        for name, value in self.template.items():
            if name.startswith(exclude):
                keep[offset:offset + value.numel()] = False
            offset += value.numel()
        return keep

    def _first_suspicious(self, client: int, round_idx: int) -> int:
        """Earliest stored round from which the client's logged z stayed below -z_cut up to now (its evidence
        trail), else this round."""
        first = round_idx
        for d in reversed(self.diagnostics):
            r, z = int(d["round"]), dict(d["z"])  # type: ignore[call-overload]
            if r < round_idx - self.rewind_window or z.get(client, 0.0) >= -self.screen.z_cut:
                break
            first = r
        return first

    def _rewind_to(self, target: int) -> None:
        """Restore the chosen coordinates of the Global model and of every parent to the Global model stored at
        the start of `target` (H8, variant a: FedEraser-style rollback without replay)."""
        old = self.stored[target]
        assert self.rewind_mask is not None
        self.global_vec = torch.where(self.rewind_mask, old, self.global_vec)
        for g in self.parents:
            self.parents[g] = torch.where(self.rewind_mask, old, self.parents[g])

    def run_round(self, round_idx: int) -> None:
        start = self.global_vec.clone()
        if self.rewind != "none":
            self.stored[round_idx] = start.clone()
            for r in [r for r in self.stored if r < round_idx - self.rewind_window]:
                del self.stored[r]
        ids = list(range(len(self.clients)))
        to_global, to_parent = self._train_all(start, round_idx)

        self.evidence.release()   # clients whose sentence is served are scored again from this round
        before = set(self.evidence.isolated)
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
            if self.screen.signal in ("weights_critical", "shared_critical"):
                # Union of two screens: each signal is z-scored and gap-gated on its own, and a client's z is the
                # lower of the two, so either signal alone can exclude or isolate it (GPU batch 1: the whole update
                # and the critical layers each missed the backdoor on a different seed).
                first = self.screen.signal.removesuffix("_critical")    # "weights" or "shared" (Phase 8)
                parts = {name: self._signal_z(name, to_global, updates, scored, in_ref)
                         for name in (first, "critical")}
                raw_scores, score_stats, z_logged, z = parts[first]
                score_stats = {**score_stats, "critical": parts["critical"][1],
                               "critical_scores": parts["critical"][0]}
                for c, v in parts["critical"][3].items():
                    if v < z[c]:
                        z[c], z_logged[c] = v, parts["critical"][2][c]
                cut = -self.screen.evidence_cut
                for c in scored:
                    both = parts[first][3].get(c, 0.0) < cut and parts["critical"][3].get(c, 0.0) < cut
                    self.both_recent[c] = (self.both_recent.get(c, []) + [both])[-AGREE_ROUNDS:]
            else:
                raw_scores, score_stats, z_logged, z = self._signal_z(self.screen.signal, to_global, updates,
                                                                      scored, in_ref)
            z_evidence = dict(z)
            if self.screen.drill_down:
                parent_z = self._group_z(updates, scored, isolated, in_ref, round_idx)
                flagged_groups = {g for g, v in parent_z.items() if v < -self.screen.group_cut}
                if self.screen.drill_mode == "evidence":
                    # 4b: members sharing their flagged parent's unusual direction get that as evidence.
                    align_z = self._align(updates, scored, isolated, in_ref, flagged_groups)
                    for c, a in align_z.items():
                        z_evidence[c] = min(z[c], -a)
            self.evidence.update(z_evidence, self.two_signal() if self.screen.sentence_single else frozenset())
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
        rewind_target = None
        new_isolated = set(self.evidence.isolated) - before
        trigger = new_isolated & self.two_signal() if self.rewind_agree else new_isolated
        if self.rewind != "none" and trigger:
            # H8: the newly isolated clients' updates since their first suspicious round are in the shared model.
            # Go back to the stored model from before the earliest of those rounds instead of aggregating.
            rewind_target = max(min(self.stored), min(self._first_suspicious(c, round_idx) for c in trigger))
            self._rewind_to(rewind_target)
            self.rewound.append({"round": round_idx, "target": rewind_target})
            kept = []
        if kept and self.global_rule == "arc_nnm_tm":
            # Layer L0 under the screen: the attackers still assumed present are those not yet excluded.
            f = self.global_f if self.global_f_fixed else max(self.global_f - len(excluded),
                                                               math.ceil(self.global_f_floor * self.global_f))
            f = max(0, min(f, (len(kept) - 1) // 2))
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
                                 "regroup": event.value, "groups": dict(self.grouping.assignment),
                                 "rewind_target": rewind_target, "two_signal": sorted(self.two_signal())})

    def _signal_z(self, signal: str, to_global: dict[int, torch.Tensor], updates: list[torch.Tensor],
                  scored: list[int], in_ref: torch.Tensor
                  ) -> tuple[dict[int, float], dict[str, object], dict[int, float], dict[int, float]]:
        """One signal's scores, their statistics, the logged robust z and the z the cuts act on (one- or two-sided,
        zeroed on the bad side in rounds the gap gate does not open)."""
        if signal in ("probe_logits", "head_gap"):
            scores = self._probe_scores(to_global, scored, in_ref)
        else:
            scores = screening_scores(updates, self.global_defense.layout, self._masks(scored),
                                      replace(self.screen, signal=signal), reference_rows=in_ref)
        ref_scores = scores[in_ref]
        score_stats: dict[str, object] = {}
        if ref_scores.numel():
            med = ref_scores.median()
            mad = (ref_scores - med).abs().median()
            scale = max(1.4826 * float(mad), self.screen.relative_floor * abs(float(med)), self.screen.mad_floor)
            score_stats = {"median": float(med), "mad": float(mad), "scale": scale,
                           "relative_floor": self.screen.relative_floor * abs(float(med)),
                           "absolute_floor": self.screen.mad_floor, "reference_n": int(in_ref.sum())}
        if self.screen.shadow_reference and signal in ("weights", "shared", "critical", "root_head"):
            # Logged only (Phase 7): the same signal scored against another reference, for paired comparisons.
            shadow = screening_scores(updates, self.global_defense.layout, self._masks(scored),
                                      replace(self.screen, signal=signal, reference=self.screen.shadow_reference),
                                      reference_rows=in_ref)
            score_stats["shadow_scores"] = shadow.tolist()
            score_stats["shadow_gap_ratio"] = gap_ratio(shadow.tolist())
        raw_scores = dict(zip(scored, scores.tolist(), strict=True))
        zs = robust_z(scores, floor=self.screen.relative_floor, reference=ref_scores,
                      absolute_floor=self.screen.mad_floor)
        z_logged = {c: float(v) for c, v in zip(scored, zs.tolist(), strict=True)}
        z = one_sided(z_logged, self.screen.two_sided)
        split = gap_ratio(scores.tolist())
        score_stats["gap_ratio"] = split
        if self.screen.gap_gate > 0 and split < self.screen.gap_gate:
            z = {c: max(v, 0.0) for c, v in z.items()}   # no separate group this round: no bad evidence
        elif self.screen.two_sided and self.screen.high_gate > 0 and split < self.screen.high_gate:
            # The high side (clients unusually close to the others) only in rounds that split strongly, as with
            # colluding ALIE / Min-Max attackers; backdoor attackers and most honest outliers' flags differ (8.2).
            z = {c: (0.0 if z_logged[c] > 0 else v) for c, v in z.items()}
        return raw_scores, score_stats, z_logged, z

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
