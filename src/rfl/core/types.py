"""Data passed between components. Every layer may import this module; it imports nothing from rfl."""
from __future__ import annotations

from dataclasses import dataclass

import torch

StateDelta = dict[str, torch.Tensor]
"""Trained weights minus the weights the sender started from, keyed by parameter name."""


@dataclass(frozen=True)
class ClientData:
    """One client's preprocessed images and labels, already on the compute device."""

    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor

    @property
    def num_train(self) -> int:
        return int(self.y_train.shape[0])


@dataclass(frozen=True)
class ClientUpdate:
    """One upload. Also used for a parent's pre-aggregated update, with `client_id` set to the group id."""

    client_id: int
    num_samples: int
    delta: StateDelta
    class_mask: torch.Tensor | None = None  # bool [C]: classes present in the sender's training data


@dataclass(frozen=True)
class DetectionResult:
    """Outcome of one detector pass over a set of updates."""

    trust: dict[int, float]  # sender id -> trust this round (higher = more trusted)
    flagged: frozenset[int]  # sender ids judged malicious this round


@dataclass(frozen=True)
class RoundReport:
    """What a method reveals about one round, for the detection metrics (rfl.evaluation.detection). Methods
    without a detector report nothing flagged and no trust."""

    clients: DetectionResult  # trust: higher = more trusted, any scale; empty if the method has none
    groups: dict[int, int] | None = None  # client -> parent this round (grouped methods)
    flagged_groups: frozenset[int] | None = None  # parents the method judged malicious; None = no parent check

