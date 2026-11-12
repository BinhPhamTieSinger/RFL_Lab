"""Plain local SGD (the FedAvg client), matching the authors' `_update_standard`: a fresh SGD optimizer each
round (Nesterov momentum 0.9, weight decay 1e-4, foreach=False), batches of min(batch_size, n) shuffled with
torch.randperm on the data's device, and a cosine learning-rate schedule over rounds."""
from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from rfl.core.state import flatten
from rfl.core.types import ClientData


@dataclass(frozen=True)
class SGDSettings:
    local_epochs: int = 3
    batch_size: int = 32
    lr: float = 0.05
    min_lr: float = 0.001
    momentum: float = 0.9
    nesterov: bool = True
    weight_decay: float = 1e-4


def cosine_lr(round_idx: int, total_rounds: int, lr: float, min_lr: float) -> float:
    """Authors' `get_current_lr`: `lr` in round 1, `min_lr` in the last round (rounds count from 1)."""
    if total_rounds <= 1:
        return lr
    return min_lr + (lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * (round_idx - 1) / (total_rounds - 1)))


def batches(data: ClientData, targets: torch.Tensor | None, epochs: int, batch_size: int
            ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Shuffled mini-batches for `epochs` passes (a fresh torch.randperm per pass, like the authors' loader).
    `targets` replaces the labels (a label-flipping attacker)."""
    n = data.num_train
    y_all = data.y_train if targets is None else targets
    batch = max(1, min(batch_size, n))
    for _ in range(epochs):
        order = torch.randperm(n, device=data.x_train.device)
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            yield data.x_train[idx], y_all[idx]


class LocalRule:
    """Base for single-model local rules. Defaults: the client uploads its whole trained state (flat, see
    rfl.core.state), keeps nothing between rounds and is scored with the global model."""

    def __init__(self, settings: SGDSettings, total_rounds: int) -> None:
        self.s = settings
        self.total_rounds = total_rounds

    def lr(self, round_idx: int) -> float:
        return cosine_lr(round_idx, self.total_rounds, self.s.lr, self.s.min_lr)

    def sgd(self, params: Iterable[torch.nn.Parameter], lr: float) -> torch.optim.SGD:
        return torch.optim.SGD(params, lr=lr, momentum=self.s.momentum, nesterov=self.s.nesterov,
                               weight_decay=self.s.weight_decay, foreach=False)

    def train(self, model: torch.nn.Module, data: ClientData, round_idx: int, client_id: int,
              targets: torch.Tensor | None = None) -> None:
        raise NotImplementedError

    def upload(self, model: torch.nn.Module, client_id: int) -> torch.Tensor:
        return flatten(model.state_dict())

    def end_round(self) -> None:
        """Called once after all clients trained (SCAFFOLD updates its server control variate here)."""

    def personal_model(self, client_id: int) -> torch.Tensor | None:
        """Flat state this client is scored with; None means the global model."""
        return None


class SGDTrainer(LocalRule):
    """FedAvg's client."""

    def train(self, model: torch.nn.Module, data: ClientData, round_idx: int, client_id: int,
              targets: torch.Tensor | None = None) -> None:
        if data.num_train == 0:
            return
        opt = self.sgd(model.parameters(), self.lr(round_idx))
        model.train()
        for x, y in batches(data, targets, self.s.local_epochs, self.s.batch_size):
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(x), y).backward()
            opt.step()
