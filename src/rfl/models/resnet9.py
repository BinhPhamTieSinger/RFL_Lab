"""ResNet-9 with base width 32, as in the authors' `src/core/model.py`. Module names and creation order match
theirs, so state dicts are interchangeable and the same seed gives the same initial weights (tested)."""
from __future__ import annotations

import torch
import torch.nn as nn


def conv_block(c_in: int, c_out: int, pool: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                               nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class _Backbone(nn.Module):
    """The shared ResNet-9 trunk. Subclasses add their heads after calling `__init__`, so creation order (and
    therefore seeded initialization) matches the authors' models."""

    def __init__(self, in_channels: int, base_channels: int) -> None:
        super().__init__()
        c = base_channels
        self.prep = conv_block(in_channels, c)
        self.layer1 = conv_block(c, c * 2, pool=True)
        self.res1 = nn.Sequential(conv_block(c * 2, c * 2), conv_block(c * 2, c * 2))
        self.layer2 = conv_block(c * 2, c * 4, pool=True)
        self.layer3 = conv_block(c * 4, c * 8, pool=True)
        self.res2 = nn.Sequential(conv_block(c * 8, c * 8), conv_block(c * 8, c * 8))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer1(self.prep(x))
        out = self.res1(out) + out
        out = self.layer3(self.layer2(out))
        out = self.res2(out) + out
        return torch.flatten(self.pool(out), 1)


class ResNet9(_Backbone):
    """Single-head ResNet-9 used by FedAvg, FedProx, SCAFFOLD, Multi-Krum, Ditto and FedRep."""

    def __init__(self, in_channels: int = 3, num_classes: int = 100, base_channels: int = 32) -> None:
        super().__init__(in_channels, base_channels)
        self.fc2 = nn.Linear(base_channels * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.extract_features(x))


class MultiHeadResNet9(_Backbone):
    """FedHEP's model: one trunk, three linear heads. `fc2_root` is shared by everyone, `fc2_parent` by a group,
    `fc2_local` is kept by each client. Matches the authors' `MultiHeadResNet9`."""

    def __init__(self, in_channels: int = 3, num_classes: int = 100, base_channels: int = 32) -> None:
        super().__init__(in_channels, base_channels)
        self.fc2_root = nn.Linear(base_channels * 8, num_classes)
        self.fc2_parent = nn.Linear(base_channels * 8, num_classes)
        self.fc2_local = nn.Linear(base_channels * 8, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (root, parent, local) logits."""
        h = self.extract_features(x)
        return self.fc2_root(h), self.fc2_parent(h), self.fc2_local(h)
