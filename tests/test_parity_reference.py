"""Parity with the authors' implementation (the reference). Each test calls their code directly and requires an
identical result. Skipped when the reference folder isn't present; set RFL_REFERENCE to its path, default
../Topology-aware-FDL next to this repo."""
import importlib
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from rfl.client.sgd import cosine_lr
from rfl.data.cifar100 import normalize, seeded_subsets
from rfl.data.partition import DirichletPartitioner, IIDPartitioner
from rfl.models.resnet9 import ResNet9

REF = Path(os.environ.get("RFL_REFERENCE", Path(__file__).resolve().parents[2] / "Topology-aware-FDL"))
pytestmark = pytest.mark.skipif(not (REF / "src" / "data" / "dataset.py").exists(),
                                reason=f"reference code not found at {REF}")


@pytest.fixture(scope="module")
def ref():
    sys.path.insert(0, str(REF))
    try:
        yield types.SimpleNamespace(
            dataset=importlib.import_module("src.data.dataset"),
            model=importlib.import_module("src.core.model"),
            base_engine=importlib.import_module("src.core.base_engine"),
        )
    finally:
        sys.path.remove(str(REF))


class _Labels:
    """Minimal stand-in for a dataset: what the authors' partition_data reads."""

    def __init__(self, y):
        self.targets = y

    def __len__(self):
        return len(self.targets)


@pytest.mark.parametrize("seed", [42, 123, 7])
@pytest.mark.parametrize("alpha", [1.0, 0.5, 0.1, 0.05])
def test_dirichlet_partition_identical(ref, seed, alpha):
    y = np.random.RandomState(seed + 1).randint(0, 100, 10000)
    theirs = ref.dataset.partition_data(_Labels(y), 15, non_iid=True, alpha=alpha, seed=seed)
    ours = DirichletPartitioner(alpha).split(y, 15, seed)
    for c in range(15):
        assert ours[c].tolist() == theirs[c]


@pytest.mark.parametrize("seed", [42, 123, 7])
def test_iid_partition_identical(ref, seed):
    y = np.random.RandomState(seed).randint(0, 100, 10000)
    theirs = ref.dataset.partition_data(_Labels(y), 15, non_iid=False, alpha=1.0, seed=seed)
    ours = IIDPartitioner().split(y, 15, seed)
    for c in range(15):
        assert ours[c].tolist() == theirs[c]


@pytest.mark.parametrize("seed", [42, 123, 7])
def test_cifar_subsets_identical(ref, seed, monkeypatch):
    """Run the authors' get_cifar100 against a fake CIFAR100 and compare the subset indices they keep."""
    class FakeCIFAR(torch.utils.data.Dataset):
        def __init__(self, root, train, download, transform):
            self.n = 50000 if train else 10000

        def __len__(self):
            return self.n

    ds = ref.dataset
    monkeypatch.setattr(ds.datasets, "CIFAR100", FakeCIFAR)
    monkeypatch.setattr(ds, "_auto_link_dataset", lambda *a, **k: False)
    monkeypatch.setattr(ds, "_acquire_download_lock", lambda *a, **k: None)
    monkeypatch.setattr(ds, "_release_download_lock", lambda *a, **k: None)
    monkeypatch.setattr(ds, "_resolve_safe_data_dir", lambda d: d)
    tr_sub, te_sub = ds.get_cifar100("unused", train_subset=10000, test_subset=3000, seed=seed)
    tr, te = seeded_subsets(50000, 10000, 10000, 3000, seed)
    assert list(tr_sub.indices) == tr.tolist() and list(te_sub.indices) == te.tolist()


def test_normalization_identical_to_their_transform(ref):
    tv = ref.dataset.transforms
    t = tv.Compose([tv.ToTensor(), tv.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    imgs = np.random.RandomState(0).randint(0, 256, (4, 32, 32, 3), dtype=np.uint8)
    theirs = torch.stack([t(im) for im in imgs])
    assert torch.allclose(normalize(imgs), theirs, atol=1e-6)


def test_resnet9_same_structure_and_same_init(ref):
    torch.manual_seed(42)
    theirs = ref.model.ResNet9(in_channels=3, num_classes=100).state_dict()
    torch.manual_seed(42)
    ours = ResNet9(3, 100).state_dict()
    assert list(ours) == list(theirs)
    for k in ours:
        assert torch.equal(ours[k], theirs[k]), k


def test_learning_rate_schedule_identical(ref):
    cfg = types.SimpleNamespace(clients=types.SimpleNamespace(local_lr=0.05, min_lr=0.001), num_rounds=25)
    fake_engine = types.SimpleNamespace(config=cfg)
    for r in range(1, 26):
        theirs = ref.base_engine.BaseEngine.get_current_lr(fake_engine, r)
        assert cosine_lr(r, 25, 0.05, 0.001) == pytest.approx(theirs)
