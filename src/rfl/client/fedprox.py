"""FedProx (Li et al., 2020), ported from the authors' `FedProxUpdater._update_fedprox`
(src/baselines/fedprox_updater.py): plain local SGD plus (mu / 2) ||w - w_received||^2 over the parameters."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from rfl.client.sgd import LocalRule, SGDSettings, batches
from rfl.core.types import ClientData


class FedProxTrainer(LocalRule):
    def __init__(self, settings: SGDSettings, total_rounds: int, mu: float = 0.01) -> None:
        super().__init__(settings, total_rounds)
        self.mu = mu

    def train(self, model: torch.nn.Module, data: ClientData, round_idx: int, client_id: int,
              targets: torch.Tensor | None = None) -> None:
        if data.num_train == 0:
            return
        anchor = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
        opt = self.sgd(model.parameters(), self.lr(round_idx))
        model.train()
        for x, y in batches(data, targets, self.s.local_epochs, self.s.batch_size):
            opt.zero_grad(set_to_none=True)
            w = torch.nn.utils.parameters_to_vector(model.parameters())
            loss = F.cross_entropy(model(x), y) + 0.5 * self.mu * torch.sum((w - anchor) ** 2)
            loss.backward()
            opt.step()
