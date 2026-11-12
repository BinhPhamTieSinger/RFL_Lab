"""Attack success rate of a backdoor: the share of honest clients' test images, of classes other than the
target, that are predicted as the target once the trigger is stamped in. Pooled over clients."""
from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from rfl.core.types import ClientData


@torch.no_grad()
def attack_success_rate(predict: Callable[[int, torch.Tensor], torch.Tensor], clients: Sequence[ClientData],
                        trigger: Callable[[torch.Tensor], torch.Tensor], target: int,
                        exclude: frozenset[int] = frozenset(), chunk: int = 1024) -> float | None:
    hits = total = 0
    for cid, c in enumerate(clients):
        if cid in exclude:
            continue
        x = c.x_test[c.y_test != target]
        for i in range(0, x.shape[0], chunk):
            pred = predict(cid, trigger(x[i:i + chunk])).argmax(dim=1)
            hits += int((pred == target).sum())
            total += int(pred.shape[0])
    return 100.0 * hits / total if total else None
