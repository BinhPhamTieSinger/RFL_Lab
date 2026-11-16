"""FedHEP's round, ported from the authors' `HierarchicalEnsembleEngine.run_round` (shared-backbone mode, no
defense, full participation), the configuration behind their Table 2 FedHEP rows.

Per round:
 1. every client loads the global model, its group's parent head and its own local head, and trains;
 2. the grouping may regroup clients (see rfl.grouping.similarity);
 3. each parent becomes the members' sample-weighted average, or a copy of the round-start global model when
    members agree strongly (mean cosine to their group mean > 0.5), then is smoothed with adaptive momentum;
 4. the global model becomes the sample-weighted average of all clients.

As in the authors' code, parents and the global model are full models, but clients only ever take the parent's
`fc2_parent` head; and each client is evaluated with the model it just trained, its three heads blended.
Attackers (if any) train and upload like everyone else, with their attack applied; they are not evaluated.

Server-side states are flat vectors (rfl.core.state), like the authors', because grouping and momentum act on
the full flattened update including BatchNorm buffers and batch counters."""
from __future__ import annotations

from collections.abc import Sequence

import torch

from rfl.aggregation.fedavg import FedAvg
from rfl.aggregation.momentum import adaptive_momentum_beta
from rfl.attacks.adaptive import CoordinatedAttack
from rfl.client.fedhep import FedHEPTrainer, HeadMemory, ensemble_logits, head_agreement_logits
from rfl.core.interfaces import Attack
from rfl.core.state import flatten, unflatten
from rfl.core.types import ClientData, ClientUpdate, DetectionResult, RoundReport
from rfl.defense.soft_cosine import flat_layout
from rfl.grouping.similarity import Regroup, SimilarityGrouping, mean_member_similarity
from rfl.models.resnet9 import MultiHeadResNet9


class FedHEPServer:
    def __init__(self, model: MultiHeadResNet9, clients: Sequence[ClientData], trainer: FedHEPTrainer,
                 grouping: SimilarityGrouping, sync_similarity: float = 0.5, attack: Attack | None = None,
                 attackers: frozenset[int] = frozenset(), similarity_on: str = "full") -> None:
        self.model = model
        self.clients = clients
        self.trainer = trainer
        self.grouping = grouping
        self.sync_similarity = sync_similarity
        self.aggregator = FedAvg()
        self.attack = attack
        self.attackers = attackers if attack is not None else frozenset()
        self.attack_stop = 0  # attackers train honestly after this round (0 = never); set by the runner
        self.unscored = self.attackers  # attackers are not evaluated, as in the authors' evaluate_ensemble
        self.template = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.global_vec = flatten(self.template)
        # "full": grouping, agreement and momentum compare whole flat updates, BatchNorm counters included
        # (authors). "weights": trained weights only (Phase 3 fix: the counters swamp the comparison).
        if similarity_on not in ("full", "weights"):
            raise ValueError(f"Unknown similarity_on {similarity_on!r}")
        self.similarity_mask = (None if similarity_on == "full" else
                                ~flat_layout(self.template, {k for k, _ in model.named_parameters()}).buffer_mask)
        self.parents = {g: self.global_vec.clone() for g in range(grouping.num_groups)}
        self.memory = [HeadMemory() for _ in clients]
        self.trained: list[torch.Tensor] = [self.global_vec.clone() for _ in clients]  # each client's last model
        self.overruled = self.seen = 0   # head-agreement inference counters (predict_agree)
        self.report: RoundReport | None = None
        self.diagnostics: list[dict[str, object]] = []  # per round: grouping event, agreement, parent rule used

    def _for_similarity(self, deltas: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        return deltas if self.similarity_mask is None else {c: d[self.similarity_mask] for c, d in deltas.items()}

    def _load(self, vec: torch.Tensor) -> None:
        self.model.load_state_dict(unflatten(vec, self.template))

    def _parent_head(self, group: int) -> dict[str, torch.Tensor]:
        self._load(self.parents[group])
        return {k: v.clone() for k, v in self.model.fc2_parent.state_dict().items()}

    def _train_all(self, start: torch.Tensor, round_idx: int
                   ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        """Every client trains from `start` plus its group's parent head and its own local head. Returns the
        uploads for Global and for the parent: the trained model, or for an attacker the poisoned one (random
        noise draws separately for the two, as in the authors' code)."""
        parent_heads = {g: self._parent_head(g) for g in self.parents}
        to_global, to_parent = {}, {}
        for cid, data in enumerate(self.clients):
            mem = self.memory[cid]
            self._load(start)
            self.model.fc2_parent.load_state_dict(parent_heads[self.grouping.assignment[cid]])
            if mem.local_head is not None:
                self.model.fc2_local.load_state_dict(mem.local_head)
            attacking = self.attack_stop == 0 or round_idx <= self.attack_stop
            bad = self.attack if cid in self.attackers and attacking else None
            data = bad.poison_data(cid, data) if bad else data
            targets = bad.poison_labels(data.y_train, self.trainer.num_classes) if bad else None
            self.trainer.train(self.model, data, round_idx, mem, targets)
            self.trained[cid] = flatten(self.model.state_dict())
            to_global[cid] = bad.poison_upload(self.trained[cid], start) if bad else self.trained[cid]
            to_parent[cid] = bad.poison_upload(self.trained[cid], start) if bad else self.trained[cid]
        if isinstance(self.attack, CoordinatedAttack) and (self.attack_stop == 0 or round_idx <= self.attack_stop):
            mask = ~flat_layout(self.template, {k for k, _ in self.model.named_parameters()}).buffer_mask
            to_global = self.attack.poison_batch(to_global, start, self.attackers, mask)
            to_parent = self.attack.poison_batch(to_parent, start, self.attackers, mask)
        return to_global, to_parent

    def run_round(self, round_idx: int) -> None:
        start = self.global_vec.clone()
        ids = list(range(len(self.clients)))
        to_global, to_parent = self._train_all(start, round_idx)
        deltas = {c: to_global[c] - start for c in ids}
        parent_deltas = {c: to_parent[c] - start for c in ids}

        sim = self._for_similarity(deltas)
        event = self.grouping.step(sim, round_idx)
        if event is Regroup.INITIAL:
            self.parents = {g: start.clone() for g in self.parents}
            for cid in ids:
                self.memory[cid].local_head = None

        agree = mean_member_similarity(self.grouping.assignment, sim, self.grouping.num_groups)
        for g in self.parents:
            members = self.grouping.members(g, ids)
            if not members:
                continue
            new = start.clone() if agree > self.sync_similarity else start + self._average(members, parent_deltas)
            beta = adaptive_momentum_beta([sim[c] for c in members])
            self.parents[g] = beta * self.parents[g] + (1.0 - beta) * new if beta > 0 else new

        self.global_vec = start + self._average(ids, deltas)
        self.report = RoundReport(DetectionResult({}, frozenset()), groups=dict(self.grouping.assignment))
        self.diagnostics.append({"round": round_idx, "regroup": event.value, "member_agreement": agree,
                                 "parents_copy_global": agree > self.sync_similarity,
                                 "groups": dict(self.grouping.assignment)})

    def _average(self, members: Sequence[int], deltas: dict[int, torch.Tensor]) -> torch.Tensor:
        updates = [ClientUpdate(c, self.clients[c].num_train, {"state": deltas[c]}) for c in members]
        return self.aggregator.aggregate(updates)["state"]

    @torch.no_grad()
    def predict(self, client_id: int, images: torch.Tensor) -> torch.Tensor:
        self._load(self.trained[client_id])
        self.model.eval()
        return ensemble_logits(self.model, images, self.memory[client_id])

    @torch.no_grad()
    def predict_agree(self, client_id: int, images: torch.Tensor) -> torch.Tensor:
        """Head-agreement inference (idea 9); counts overruled inputs in `self.overruled` and `self.seen`."""
        self._load(self.trained[client_id])
        self.model.eval()
        z, n = head_agreement_logits(self.model, images, self.memory[client_id])
        self.overruled += n
        self.seen += images.shape[0]
        return z

    @torch.no_grad()
    def predict_head(self, head: str, client_id: int, images: torch.Tensor) -> torch.Tensor:
        """One head's logits ("root", "parent" or "local") on the client's model, no ensemble and no class mask.
        Used to locate a backdoor: which head carries it."""
        self._load(self.trained[client_id])
        self.model.eval()
        z_root, z_parent, z_local = self.model(images)
        return {"root": z_root, "parent": z_parent, "local": z_local}[head]
