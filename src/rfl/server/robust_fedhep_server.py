"""Phase 5: Defended FedHEP with its Global step replaced by an established robust rule (rfl.defense.classic),
to compare those rules with FedHEP+'s screening on the same model, personalisation and attacks.

Per round:
 1. every client trains (as in FedHEP); non-finite uploads are banned for good (Defended FedHEP's sentinel);
 2. Global = start + rule(weight updates of the senders); BatchNorm buffers by coordinate median, as SCCF does;
 3. parents aggregate the members the rule accepted with SCCF, as Defended FedHEP does (rules that keep everyone,
    median and trimmed mean, leave all members in). FedHEP+ likewise lets one decision, screening, pick who
    enters both tiers.
FLTrust's server update comes from training a copy of the round-start model on the root set with the clients'
own trainer; the random state is saved and restored around it, so client training draws the same shuffles as
in every other method."""
from __future__ import annotations

from collections.abc import Sequence

import torch

from rfl.client.fedhep import FedHEPTrainer, HeadMemory
from rfl.core.interfaces import Attack
from rfl.core.state import flatten
from rfl.core.types import ClientData, DetectionResult, RoundReport
from rfl.defense.classic import (
    RuleResult,
    arc_nnm_trimmed_mean_rule,
    bulyan_rule,
    flame_rule,
    fltrust_rule,
    median_rule,
    trimmed_mean_rule,
)
from rfl.defense.soft_cosine import coordinate_median
from rfl.grouping.similarity import SimilarityGrouping
from rfl.models.resnet9 import MultiHeadResNet9
from rfl.server.defended_fedhep_server import DefendedFedHEPServer

RULES = ("median", "trimmed_mean", "bulyan", "flame", "fltrust", "arc_nnm_tm")


class RobustFedHEPServer(DefendedFedHEPServer):
    def __init__(self, model: MultiHeadResNet9, clients: Sequence[ClientData], trainer: FedHEPTrainer,
                 grouping: SimilarityGrouping, rule: str, assumed_attackers: int = 3, flame_lambda: float = 0.001,
                 flame_on: str = "models", root: ClientData | None = None, seed: int = 0,
                 attack: Attack | None = None, attackers: frozenset[int] = frozenset(),
                 momentum: float = 0.0) -> None:
        super().__init__(model, clients, trainer, grouping, attack=attack, attackers=attackers)
        if rule not in RULES:
            raise ValueError(f"Unknown rule {rule!r}; choose from {RULES}")
        if rule == "fltrust" and root is None:
            raise ValueError("FLTrust needs a root dataset")
        if flame_on not in ("models", "updates"):
            raise ValueError(f"Unknown flame_on {flame_on!r}")
        self.rule, self.f, self.flame_lambda, self.flame_on = rule, assumed_attackers, flame_lambda, flame_on
        self.root, self.root_memory, self.seed = root, HeadMemory(), seed
        self.weights = ~self.global_defense.layout.buffer_mask.to(self.global_vec.device)
        self.banned: set[int] = set()
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        self.momentum = momentum
        self.client_momentum: dict[int, torch.Tensor] = {}   # worker momentum m_i, kept by the server

    def _server_update(self, start: torch.Tensor, round_idx: int) -> torch.Tensor:
        assert self.root is not None
        device = self.root.x_train.device
        cpu_state = torch.get_rng_state()
        dev_state = (torch.mps.get_rng_state() if device.type == "mps" else
                     torch.cuda.get_rng_state(device) if device.type == "cuda" else
                     torch.xpu.get_rng_state(device) if device.type == "xpu" else None)
        self._load(start)
        if self.root_memory.local_head is not None:
            self.model.fc2_local.load_state_dict(self.root_memory.local_head)
        self.trainer.train(self.model, self.root, round_idx, self.root_memory)
        torch.set_rng_state(cpu_state)
        if dev_state is not None and device.type == "mps":
            torch.mps.set_rng_state(dev_state)
        elif dev_state is not None and device.type == "xpu":
            torch.xpu.set_rng_state(dev_state, device)
        elif dev_state is not None:
            torch.cuda.set_rng_state(dev_state, device)
        return flatten(self.model.state_dict()) - start

    def _apply(self, stack: torch.Tensor, start: torch.Tensor, round_idx: int) -> RuleResult:
        w = stack[:, self.weights]
        if self.rule == "median":
            return median_rule(w)
        if self.rule == "trimmed_mean":
            return trimmed_mean_rule(w, self.f)
        if self.rule == "bulyan":
            return bulyan_rule(w, self.f)
        if self.rule == "arc_nnm_tm":
            return arc_nnm_trimmed_mean_rule(w, self.f)
        if self.rule == "flame":
            gen = torch.Generator().manual_seed(self.seed * 1000 + round_idx)
            return flame_rule(w, start[self.weights], self.flame_lambda, gen, on=self.flame_on)
        return fltrust_rule(w, self._server_update(start, round_idx)[self.weights])

    def run_round(self, round_idx: int) -> None:
        start = self.global_vec.clone()
        ids = list(range(len(self.clients)))
        to_global, to_parent = self._train_all(start, round_idx)
        for c in ids:
            if not (torch.isfinite(to_global[c]).all() and torch.isfinite(to_parent[c]).all()):
                self.banned.add(c)
        senders = [c for c in ids if c not in self.banned]

        scores: dict[int, float] = {}
        accepted = senders
        if senders:
            stack = torch.stack([to_global[c] - start for c in senders])
            if self.momentum > 0:   # m_i <- beta m_i + (1 - beta) delta_i; rules see m_i (Karimireddy et al. 2021)
                for i, c in enumerate(senders):
                    prev = self.client_momentum.get(c)
                    m = stack[i] if prev is None else self.momentum * prev + (1 - self.momentum) * stack[i]
                    self.client_momentum[c] = m
                stack = torch.stack([self.client_momentum[c] for c in senders])
            result = self._apply(stack, start, round_idx)
            step = torch.empty_like(start)
            step[self.weights] = result.update.to(step.dtype)
            step[~self.weights] = coordinate_median(stack[:, ~self.weights])
            self.global_vec = start + step
            if result.accepted is not None:
                accepted = [senders[i] for i in result.accepted]
            scores = {c: float(s) for c, s in zip(senders, result.scores.tolist(), strict=True)}

        for g in self.parents:
            members = self.grouping.members(g, accepted)
            if members:
                ref = self.parents[g]
                step_p, _ = self.parent_defense.aggregate([to_parent[c] - ref for c in members],
                                                          self._masks(members))
                self.parents[g] = ref + step_p

        flagged = frozenset(self.banned | (set(senders) - set(accepted)))
        trust = {**{c: -99.0 for c in self.banned}, **scores}
        self.report = RoundReport(DetectionResult(trust, flagged), groups=dict(self.grouping.assignment))
        self.diagnostics.append({"round": round_idx, "attackers": sorted(self.attackers), "rule": self.rule,
                                 "accepted": sorted(accepted), "scores": scores, "banned": sorted(self.banned)})
