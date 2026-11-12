"""Client partitioners, ported line for line from the authors' `partition_data` (src/data/dataset.py) so the
same seed gives exactly the same split. Parity is checked in tests/test_parity_reference.py."""
from __future__ import annotations

import numpy as np


class IIDPartitioner:
    """Shuffle, then deal equal shares. The remainder (num_samples mod num_clients) is dropped, as in the
    authors' code: 10 of 10,000 CIFAR-100 training images with 15 clients."""

    def split(self, labels: np.ndarray, num_clients: int, seed: int) -> list[np.ndarray]:
        rng = np.random.RandomState(seed)
        indices = np.arange(len(labels))
        rng.shuffle(indices)
        per_client = len(labels) // num_clients
        return [indices[i * per_client : (i + 1) * per_client] for i in range(num_clients)]


class DirichletPartitioner:
    """Label skew (Hsu et al., 2019). One draw of class proportions Dir(alpha) for all classes, then each class's
    shuffled samples are cut by those proportions. Floor leftovers go round-robin from client 0."""

    def __init__(self, alpha: float) -> None:
        if alpha <= 0:
            raise ValueError("Dirichlet alpha must be positive")
        self.alpha = alpha

    def split(self, labels: np.ndarray, num_clients: int, seed: int) -> list[np.ndarray]:
        rng = np.random.RandomState(seed)
        classes = np.unique(labels)
        proportions = rng.dirichlet(self.alpha * np.ones(num_clients), len(classes))
        parts: list[list[int]] = [[] for _ in range(num_clients)]
        for k, c in enumerate(classes):
            idx = np.where(labels == c)[0].copy()
            rng.shuffle(idx)
            counts = np.floor(proportions[k] * len(idx)).astype(int)
            for j in range(len(idx) - int(counts.sum())):
                counts[j % num_clients] += 1
            for client, chunk in enumerate(np.split(idx, np.cumsum(counts)[:-1])):
                parts[client].extend(chunk.tolist())
        return [np.asarray(p, dtype=np.int64) for p in parts]


def make_partitioner(alpha: float | None) -> IIDPartitioner | DirichletPartitioner:
    """`alpha=None` means IID."""
    return IIDPartitioner() if alpha is None else DirichletPartitioner(alpha)
