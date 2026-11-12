"""The four Byzantine attacks of the authors' attack table (Table 3), ported from `PyTorchLocalUpdater` in their
src/core/updater.py (`_flip_labels`, `_apply_byzantine_attack`), and their rule for choosing attackers
(`BaseEngine.__init__`).

* label_flip: trains on label C-1-y instead of y (the client's label statistics still use the true labels);
* sign_flip: uploads start - 1.5 * update;
* gradient_ascent: uploads start - 5 * update;
* random_noise: uploads start + N(0, 2^2) noise, a fresh draw for each uploaded model."""
from __future__ import annotations

import numpy as np
import torch

from rfl.core.types import ClientData

KINDS = ("label_flip", "sign_flip", "gradient_ascent", "random_noise")


def select_attackers(num_clients: int, rate: float, seed: int) -> frozenset[int]:
    """Each client is an attacker with probability `rate`, so the count varies by seed (Binomial). Same draws as
    the authors: right after seeding NumPy, `RandomState(randint(0, 2^31 - 1))`, one `rand()` per client."""
    rng = np.random.RandomState(np.random.RandomState(seed).randint(0, 2**31 - 1))
    return frozenset(c for c in range(num_clients) if rng.rand() < rate)


def select_attackers_fixed(num_clients: int, count: int, seed: int) -> frozenset[int]:
    """Exactly `count` attackers, drawn uniformly from RandomState(seed)."""
    if not 0 <= count <= num_clients:
        raise ValueError(f"attacker count must be in [0, {num_clients}]")
    return frozenset(int(c) for c in np.random.RandomState(seed).choice(num_clients, count, replace=False))


class ByzantineAttack:
    def __init__(self, kind: str) -> None:
        if kind not in KINDS:
            raise ValueError(f"Unknown attack {kind!r}; choose from {KINDS}")
        self.kind = kind

    def poison_data(self, client_id: int, data: ClientData) -> ClientData:
        return data

    def poison_labels(self, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        return (num_classes - 1) - labels if self.kind == "label_flip" else labels

    def poison_upload(self, trained: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
        """`trained` and `start` are flat state vectors (rfl.core.state)."""
        delta = trained - start
        if self.kind == "sign_flip":
            return start - delta * 1.5
        if self.kind == "gradient_ascent":
            return start - delta * 5.0
        if self.kind == "random_noise":
            return start + torch.randn_like(start) * 2.0
        return trained
