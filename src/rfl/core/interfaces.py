"""The contracts every pluggable mechanism implements.

A new mechanism (an aggregator, a detector, a grouping rule, an attack...) is one class that satisfies one of these
protocols. The server composes them; it contains no algorithm logic of its own. See docs/ARCHITECTURE.md.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np
import torch

from rfl.core.types import ClientData, ClientUpdate, DetectionResult, StateDelta


class Partitioner(Protocol):
    """Splits a labelled dataset across clients (IID, Dirichlet label skew, ...). Returns one index array per
    client, as positions into `labels`. The same seed must always give the same split."""

    def split(self, labels: np.ndarray, num_clients: int, seed: int) -> list[np.ndarray]: ...


class ClientTrainer(Protocol):
    """A single-model local rule (plain SGD, FedProx, SCAFFOLD, Ditto, FedRep). `train` trains `model` in place
    (`round_idx` starts at 1; `targets`, if given, replace the labels); `upload` returns what the client sends, as
    a flat state (rfl.core.state); `end_round` runs once after all clients trained; `personal_model` is the flat
    state a client is scored with, or None for the global model. Base class: rfl.client.sgd.LocalRule."""

    def train(self, model: torch.nn.Module, data: ClientData, round_idx: int, client_id: int,
              targets: torch.Tensor | None = None) -> None: ...

    def upload(self, model: torch.nn.Module, client_id: int) -> torch.Tensor: ...

    def end_round(self) -> None: ...

    def personal_model(self, client_id: int) -> torch.Tensor | None: ...


class Attack(Protocol):
    """A malicious client's behaviour. Data attacks change the training data (`poison_data`, e.g. a backdoor) or
    only the training targets (`poison_labels`, which keeps the client's label statistics honest, as the authors'
    label flipping does); update attacks change what is uploaded (`trained` and `start` are flat state vectors,
    see rfl.core.state)."""

    def poison_data(self, client_id: int, data: ClientData) -> ClientData: ...

    def poison_labels(self, labels: torch.Tensor, num_classes: int) -> torch.Tensor: ...

    def poison_upload(self, trained: torch.Tensor, start: torch.Tensor) -> torch.Tensor: ...


class Grouper(Protocol):
    """Assigns clients to parents (random, by update similarity, by label distribution, ...)."""

    def assign(self, updates: Sequence[ClientUpdate], round_idx: int) -> dict[int, int]: ...


class Detector(Protocol):
    """Scores senders for maliciousness. The same contract serves both levels: clients inside a parent, and
    parents seen by Global (where each update is a parent's pre-aggregate)."""

    def inspect(self, updates: Sequence[ClientUpdate], round_idx: int) -> DetectionResult: ...


class Aggregator(Protocol):
    """Combines updates into one delta. `weights` (sender id -> weight) overrides the default weighting."""

    def aggregate(
        self, updates: Sequence[ClientUpdate], weights: Mapping[int, float] | None = None
    ) -> StateDelta: ...


class FederatedServer(Protocol):
    """A whole method as the experiment runner sees it: run one round, then predict for any client (the model a
    client is scored with is the method's choice: the global model, a personalised one, an ensemble...)."""

    def run_round(self, round_idx: int) -> None: ...

    def predict(self, client_id: int, images: torch.Tensor) -> torch.Tensor: ...
