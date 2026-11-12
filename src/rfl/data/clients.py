"""Turns a dataset and a partitioner into per-client train/test sets.

As in the authors' engine, train and test pools are partitioned separately with the same partitioner and seed,
so a client's test set follows the same class proportions as its training set."""
from __future__ import annotations

import numpy as np
import torch

from rfl.core.interfaces import Partitioner
from rfl.core.types import ClientData


def build_clients(x_train: torch.Tensor, y_train: torch.Tensor, x_test: torch.Tensor, y_test: torch.Tensor,
                  partitioner: Partitioner, num_clients: int, seed: int) -> list[ClientData]:
    tr_parts = partitioner.split(y_train.cpu().numpy(), num_clients, seed)
    te_parts = partitioner.split(y_test.cpu().numpy(), num_clients, seed)
    device = x_train.device

    def take(x: torch.Tensor, y: torch.Tensor, idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        i = torch.as_tensor(idx, dtype=torch.long, device=device)
        return x[i], y[i]

    clients = []
    for tr, te in zip(tr_parts, te_parts, strict=True):
        xtr, ytr = take(x_train, y_train, tr)
        xte, yte = take(x_test, y_test, te)
        clients.append(ClientData(xtr, ytr, xte, yte))
    return clients
