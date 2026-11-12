"""Per-client accuracy and the summary metrics.

`pooled` is the number the authors report as mean accuracy: correct predictions over all clients' test images
divided by the total count (a sample-weighted mean). `mean_client` is the plain average over clients, which is
what the paper's text describes. Both are recorded so either can be compared."""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from rfl.core.types import ClientData


@dataclass(frozen=True)
class AccuracyReport:
    pooled: float
    mean_client: float
    bottom10: float  # mean of the lowest ceil(10% of clients) accuracies, as in the authors' code
    per_client: dict[int, float]


def bottom_fraction_mean(values: Sequence[float], fraction: float = 0.1) -> float:
    ordered = sorted(values)
    k = max(1, math.ceil(fraction * len(ordered)))
    return sum(ordered[:k]) / k


@torch.no_grad()
def evaluate_clients(predict: Callable[[int, torch.Tensor], torch.Tensor], clients: Sequence[ClientData],
                     chunk: int = 1024, exclude: frozenset[int] = frozenset()) -> AccuracyReport:
    """`predict(client_id, images)` returns logits; it decides which model a client is scored with. Clients in
    `exclude` (attackers) are not scored, as in the authors' code."""
    per_client: dict[int, float] = {}
    correct_all = total_all = 0
    for cid, c in enumerate(clients):
        n = int(c.y_test.shape[0])
        if n == 0 or cid in exclude:
            continue
        correct = 0
        for s in range(0, n, chunk):
            correct += int((predict(cid, c.x_test[s:s + chunk]).argmax(1) == c.y_test[s:s + chunk]).sum())
        per_client[cid] = 100.0 * correct / n
        correct_all += correct
        total_all += n
    if total_all == 0:
        raise ValueError("No client has test data")
    accs = list(per_client.values())
    return AccuracyReport(pooled=100.0 * correct_all / total_all, mean_client=sum(accs) / len(accs),
                          bottom10=bottom_fraction_mean(accs), per_client=per_client)
