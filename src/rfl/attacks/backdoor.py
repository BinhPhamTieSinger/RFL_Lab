"""Backdoor attack (BadNets, Gu et al., 2017), optionally with model replacement (Bagdasaryan et al., 2020).
Not in the authors' code; added for the Phase 2 testbench.

An attacker stamps a small white square into the bottom-right corner of a fixed `fraction` of its training images
and relabels them `target`. It then trains normally. With `scale` > 1 it uploads start + scale * update
("model replacement": boosting the update so it survives averaging). Success is measured on honest clients' test
images of other classes, stamped with the same square (rfl.evaluation.backdoor)."""
from __future__ import annotations

import numpy as np
import torch

from rfl.core.types import ClientData

# A white pixel after the CIFAR-100 normalisation in rfl.data.cifar100: (1 - mean) / std per channel.
_WHITE = torch.tensor([(1 - 0.5071) / 0.2675, (1 - 0.4867) / 0.2565, (1 - 0.4408) / 0.2761]).view(3, 1, 1)


def stamp_trigger(images: torch.Tensor, size: int = 3) -> torch.Tensor:
    """Copy of `images` (N, 3, H, W) with the bottom-right `size` x `size` patch set to white."""
    out = images.clone()
    out[:, :, -size:, -size:] = _WHITE.to(images.device, images.dtype)
    return out


class BackdoorAttack:
    def __init__(self, target: int = 0, fraction: float = 0.5, scale: float = 1.0, seed: int = 0,
                 size: int = 3) -> None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("backdoor fraction must be in (0, 1]")
        self.target, self.fraction, self.scale, self.seed, self.size = target, fraction, scale, seed, size
        self._cache: dict[int, ClientData] = {}

    def trigger(self, images: torch.Tensor) -> torch.Tensor:
        return stamp_trigger(images, self.size)

    def poison_data(self, client_id: int, data: ClientData) -> ClientData:
        """The same images are poisoned every round (chosen once per client from `seed`)."""
        if client_id not in self._cache:
            n = data.num_train
            idx = np.random.RandomState(self.seed + client_id).choice(n, int(round(self.fraction * n)),
                                                                       replace=False)
            sel = torch.as_tensor(idx, dtype=torch.long, device=data.x_train.device)
            x, y = data.x_train.clone(), data.y_train.clone()
            x[sel] = self.trigger(x[sel])
            y[sel] = self.target
            self._cache[client_id] = ClientData(x, y, data.x_test, data.y_test)
        return self._cache[client_id]

    def poison_labels(self, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        return labels

    def poison_upload(self, trained: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
        return trained if self.scale == 1.0 else start + self.scale * (trained - start)
