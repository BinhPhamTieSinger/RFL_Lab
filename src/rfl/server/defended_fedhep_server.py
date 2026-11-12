"""Defended FedHEP's round, ported from the authors' `UnifiedDefendedEngine.run_round`
(src/core/unified_defended_engine.py), the method behind their attack table (Table 3).

Differences from FedHEP (rfl.server.fedhep_server):
 * groups stay at their random initial assignment (this engine never regroups);
 * clients with non-finite uploads, or whose trust score fell below 0.25 in an earlier round or tier, are left
   out of aggregation for good (they keep training and, if honest, are still evaluated);
 * each parent becomes parent + SCCF(members' uploads - parent), with k = 2 norm bounding
   (rfl.defense.soft_cosine);
 * Global becomes start + SCCF(uploads - start), with k = 3 and hard rejection of cosine < -0.1;
 * both tiers feed the trust ledger (rfl.defense.trust_ledger), parents first.
No parent momentum and no copy-the-global rule."""
from __future__ import annotations

from collections.abc import Sequence

import torch

from rfl.client.fedhep import FedHEPTrainer
from rfl.core.interfaces import Attack
from rfl.core.types import ClientData, DetectionResult, RoundReport
from rfl.defense.soft_cosine import SoftCosineAggregator, flat_layout
from rfl.defense.trust_ledger import TrustLedger
from rfl.grouping.similarity import SimilarityGrouping
from rfl.models.resnet9 import MultiHeadResNet9
from rfl.server.fedhep_server import FedHEPServer


class DefendedFedHEPServer(FedHEPServer):
    def __init__(self, model: MultiHeadResNet9, clients: Sequence[ClientData], trainer: FedHEPTrainer,
                 grouping: SimilarityGrouping, attack: Attack | None = None,
                 attackers: frozenset[int] = frozenset()) -> None:
        super().__init__(model, clients, trainer, grouping, attack=attack, attackers=attackers)
        layout = flat_layout(self.template, {k for k, _ in model.named_parameters()})
        self.parent_defense = SoftCosineAggregator(layout, norm_bound=2.0, hard_rejection=False)
        self.global_defense = SoftCosineAggregator(layout, norm_bound=3.0, hard_rejection=True,
                                                   hard_threshold=-0.1)
        self.ledger = TrustLedger(len(clients))

    def _masks(self, members: Sequence[int]) -> torch.Tensor | None:
        masks = [self.memory[c].active_mask for c in members]
        return torch.stack([m for m in masks if m is not None]) if all(m is not None for m in masks) else None

    def run_round(self, round_idx: int) -> None:
        start = self.global_vec.clone()
        ids = list(range(len(self.clients)))
        to_global, to_parent = self._train_all(start, round_idx)

        for c in ids:  # sentinel: non-finite uploads are excluded for good
            if not (torch.isfinite(to_global[c]).all() and torch.isfinite(to_parent[c]).all()):
                self.ledger.isolated.add(c)
        active = [c for c in ids if c not in self.ledger.isolated]

        parent_trust: dict[int, float] = {}
        for g in self.parents:
            members = self.grouping.members(g, active)
            if not members:
                continue
            ref = self.parents[g]
            updates = [to_parent[c] - ref for c in members]
            step, trust = self.parent_defense.aggregate(updates, self._masks(members))
            self.parents[g] = ref + step
            scores = {c: float(t) for c, t in zip(members, trust.tolist(), strict=True)}
            self.ledger.update(scores, len(members))
            parent_trust.update(scores)

        senders = [c for c in active if c not in self.ledger.isolated]
        global_trust: dict[int, float] = {}
        if senders:
            updates = [to_global[c] - start for c in senders]
            step, trust = self.global_defense.aggregate(updates, self._masks(senders))
            self.global_vec = start + step
            global_trust = {c: float(t) for c, t in zip(senders, trust.tolist(), strict=True)}
            self.ledger.update(global_trust, len(senders))

        rejected = {c for c, t in global_trust.items() if t == 0.0}
        shares = {c: global_trust[c] * len(senders) if c in global_trust else 0.0 for c in ids}
        self.report = RoundReport(DetectionResult(shares, frozenset(self.ledger.isolated | rejected)),
                                  groups=dict(self.grouping.assignment))
        self.diagnostics.append({"round": round_idx, "attackers": sorted(self.attackers),
                                 "isolated": sorted(self.ledger.isolated), "parent_trust": parent_trust,
                                 "global_trust": global_trust, "trust_score": dict(self.ledger.score)})
