"""CIFAR-100 loading, matching the authors' `get_cifar100` + `FastDataset`:
per-channel normalization only (no augmentation), a seeded train subset drawn first and a test subset drawn
second from the same RandomState, kept in drawn order (the partition depends on that order)."""
from __future__ import annotations

import numpy as np
import torch

MEAN = (0.5071, 0.4867, 0.4408)  # authors' values
STD = (0.2675, 0.2565, 0.2761)


def seeded_subsets(n_train: int, n_test: int, train_subset: int | None, test_subset: int | None,
                   seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the train and test subsets. Both come from one RandomState: train first, then test."""
    rng = np.random.RandomState(seed)
    tr = rng.choice(n_train, train_subset, replace=False) if train_subset and train_subset < n_train \
        else np.arange(n_train)
    te = rng.choice(n_test, test_subset, replace=False) if test_subset and test_subset < n_test \
        else np.arange(n_test)
    return tr, te


def normalize(images_uint8_nhwc: np.ndarray) -> torch.Tensor:
    """uint8 [N, H, W, C] -> float32 [N, C, H, W], identical to ToTensor() + Normalize(MEAN, STD)."""
    x = torch.from_numpy(images_uint8_nhwc).permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    return (x - mean) / std


def load_cifar100(root: str, train_subset: int | None, test_subset: int | None, seed: int,
                  device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (x_train, y_train, x_test, y_test) on `device`. Downloads into `root` if missing."""
    from torchvision.datasets import CIFAR100

    train = CIFAR100(root, train=True, download=True)
    test = CIFAR100(root, train=False, download=True)
    tr, te = seeded_subsets(len(train.targets), len(test.targets), train_subset, test_subset, seed)
    y_train = torch.as_tensor(np.asarray(train.targets)[tr], dtype=torch.long)
    y_test = torch.as_tensor(np.asarray(test.targets)[te], dtype=torch.long)
    return (normalize(train.data[tr]).to(device), y_train.to(device),
            normalize(test.data[te]).to(device), y_test.to(device))


def load_root_set(root: str, train_subset: int | None, test_subset: int | None, seed: int, size: int,
                  device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """FLTrust's clean root set: `size` training images drawn uniformly from those outside the clients' train
    subset (same distribution, no client's data touched; the paper removes them from the clients instead)."""
    from torchvision.datasets import CIFAR100

    train = CIFAR100(root, train=True, download=True)
    tr, _ = seeded_subsets(len(train.targets), 10_000, train_subset, test_subset, seed)
    rest = np.setdiff1d(np.arange(len(train.targets)), tr)
    pick = np.random.RandomState(seed + 10_007).choice(rest, size, replace=False)
    y = torch.as_tensor(np.asarray(train.targets)[pick], dtype=torch.long)
    return normalize(train.data[pick]).to(device), y.to(device)
