"""Round orchestration for single-model methods: FedAvg, FedProx, SCAFFOLD, Multi-Krum, Ditto, FedRep (the
authors' `CentralizedEngine` and `ScaffoldEngine`).

Each round every client starts from the global state, trains with its local rule and uploads a flat state
(rfl.core.state; attackers upload a poisoned one); the aggregator combines the uploads' differences from the
global state; the global state moves by the result. Like the authors' code, the state includes BatchNorm running
statistics and batch counters.

Scoring follows the authors: methods with a personal model (Ditto, FedRep) score each honest client with it;
the others score every client, attackers included, with the global model."""
from __future__ import annotations

import copy
from collections.abc import Sequence

import torch

from rfl.core.interfaces import Aggregator, Attack, ClientTrainer
from rfl.core.state import flatten, unflatten
from rfl.core.types import ClientData, ClientUpdate, DetectionResult, RoundReport


class ConsensusServer:
    def __init__(self, model: torch.nn.Module, clients: Sequence[ClientData], trainer: ClientTrainer,
                 aggregator: Aggregator, attack: Attack | None = None, attackers: frozenset[int] = frozenset(),
                 personalised: bool = False, num_classes: int = 0) -> None:
        self.global_model = model
        self.work_model = copy.deepcopy(model)
        self.clients = clients
        self.trainer = trainer
        self.aggregator = aggregator
        self.attack = attack
        self.attackers = attackers if attack is not None else frozenset()
        self.num_classes = num_classes  # needed by label-flipping attackers
        self.unscored = self.attackers if personalised else frozenset()
        self.template = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.report: RoundReport | None = None

    def run_round(self, round_idx: int) -> None:
        start_state = {k: v.detach().clone() for k, v in self.global_model.state_dict().items()}
        start = flatten(start_state)
        updates: list[ClientUpdate] = []
        for cid, data in enumerate(self.clients):
            self.work_model.load_state_dict(start_state)
            bad = self.attack if cid in self.attackers else None
            data = bad.poison_data(cid, data) if bad else data
            targets = bad.poison_labels(data.y_train, self.num_classes) if bad else None
            self.trainer.train(self.work_model, data, round_idx, cid, targets)
            upload = self.trainer.upload(self.work_model, cid)
            if bad:
                upload = bad.poison_upload(upload, start)
            updates.append(ClientUpdate(cid, data.num_train, {"state": upload - start}))
        self.trainer.end_round()
        step = self.aggregator.aggregate(updates)["state"]
        self.global_model.load_state_dict(unflatten(start + step, self.template))
        # A selecting aggregator (Multi-Krum) flags the clients it leaves out; lower Krum score = more trusted.
        selected = getattr(self.aggregator, "last_selected", None)
        everyone = frozenset(range(len(self.clients)))
        flagged = everyone - frozenset(selected) if selected is not None else frozenset()
        trust = {c: -s for c, s in getattr(self.aggregator, "last_scores", {}).items()}
        self.report = RoundReport(DetectionResult(trust, flagged))

    @torch.no_grad()
    def predict(self, client_id: int, images: torch.Tensor) -> torch.Tensor:
        personal = self.trainer.personal_model(client_id)
        if personal is None:
            self.global_model.eval()
            return self.global_model(images)
        self.work_model.load_state_dict(unflatten(personal, self.template))
        self.work_model.eval()
        return self.work_model(images)
