"""Ditto (Li et al., 2021), ported from the authors' `_update_ditto` (src/core/updater.py). Each round a client
(1) trains the received global model with plain local SGD and uploads it, then (2) trains its own personal model
with CE + (lambda / 2) ||w_p - w_received||^2 over the parameters. The personal model starts, in a client's first
round, from the global model it just trained. Clients are scored with their personal model."""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from rfl.client.sgd import LocalRule, SGDSettings, batches
from rfl.core.state import flatten, unflatten
from rfl.core.types import ClientData


class DittoTrainer(LocalRule):
    def __init__(self, settings: SGDSettings, total_rounds: int, lam: float = 0.05) -> None:
        super().__init__(settings, total_rounds)
        self.lam = lam
        self.personal: dict[int, torch.Tensor] = {}
        self._uploads: dict[int, torch.Tensor] = {}

    def train(self, model: torch.nn.Module, data: ClientData, round_idx: int, client_id: int,
              targets: torch.Tensor | None = None) -> None:
        if data.num_train == 0:
            return
        lr = self.lr(round_idx)
        anchor = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
        opt = self.sgd(model.parameters(), lr)
        model.train()
        for x, y in batches(data, targets, self.s.local_epochs, self.s.batch_size):
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(x), y).backward()
            opt.step()
        self._uploads[client_id] = flatten(model.state_dict())

        own = copy.deepcopy(model)
        if client_id in self.personal:
            own.load_state_dict(unflatten(self.personal[client_id], model.state_dict()))
        own.train()
        opt = self.sgd(own.parameters(), lr)
        for x, y in batches(data, targets, self.s.local_epochs, self.s.batch_size):
            opt.zero_grad(set_to_none=True)
            w = torch.nn.utils.parameters_to_vector(own.parameters())
            loss = F.cross_entropy(own(x), y) + 0.5 * self.lam * torch.sum((w - anchor) ** 2)
            loss.backward()
            opt.step()
        self.personal[client_id] = flatten(own.state_dict())

    def upload(self, model: torch.nn.Module, client_id: int) -> torch.Tensor:
        return self._uploads.pop(client_id, flatten(model.state_dict()))

    def personal_model(self, client_id: int) -> torch.Tensor | None:
        return self.personal.get(client_id)
