import base64
import io

import numpy as np
import torch
from PIL import Image
from rfl_viz import archive
from rfl_viz.catalog import validate
from rfl_viz.partitions import describe_clients

from rfl.data.cifar100 import normalize


def test_archive_reads_results_and_incomplete_configs(tmp_path, monkeypatch):
    folder = tmp_path / "outputs" / "old-run"
    folder.mkdir(parents=True)
    (folder / "config.yaml").write_text("method: fedavg\nrun:\n  device: cuda\n")
    monkeypatch.setattr(archive, "LAB", tmp_path)
    monkeypatch.setattr(archive, "_cached", (0, {}))
    entry = archive.history(refresh=True)[0]
    assert entry["status"] == "incomplete"
    (folder / "metrics.json").write_text('{"final":{"pooled":12.5},"history":[]}')
    assert archive.history(refresh=True)[0]["status"] == "completed"
    assert archive.detail(entry["id"])["result"]["final"]["pooled"] == 12.5


def test_partition_images_are_real_and_read_only(monkeypatch):
    from types import SimpleNamespace

    import torchvision.datasets

    monkeypatch.setattr(
        torchvision.datasets, "CIFAR100", lambda *a, **k: SimpleNamespace(classes=[str(i) for i in range(100)])
    )
    pixels = np.arange(3 * 32 * 32, dtype=np.uint8).reshape(1, 32, 32, 3)
    x = normalize(pixels)
    y = torch.tensor([2])
    client = SimpleNamespace(x_train=x, y_train=y, y_test=y, num_train=1)
    state = torch.random.get_rng_state().clone()
    description = describe_clients(validate("fedhep_plus", {}), [client])[0]
    decoded = np.asarray(
        Image.open(io.BytesIO(base64.b64decode(description["samples"][0]["image"].split(",")[1])))
    )
    assert np.array_equal(decoded, pixels[0])
    assert torch.equal(x, normalize(pixels))
    assert torch.equal(state, torch.random.get_rng_state())
    assert description["histogram"][2] == 1
    assert description["r_skew"] == 0


def test_archive_serves_results_with_nan_as_null(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from rfl_viz.api import app

    folder = tmp_path / "outputs" / "nan-run"
    folder.mkdir(parents=True)
    (folder / "config.yaml").write_text("method: fedhep\n")
    (folder / "metrics.json").write_text('{"final":{"pooled":1.0},"diagnostics":[{"member_agreement":NaN}]}')
    monkeypatch.setattr(archive, "LAB", tmp_path)
    entry = archive.history(refresh=True)[0]
    response = TestClient(app).get(f"/experiments/{entry['id']}")
    assert response.status_code == 200
    assert response.json()["result"]["diagnostics"][0]["member_agreement"] is None


def test_archive_is_not_empty_soon_after_boot(tmp_path, monkeypatch):
    import importlib

    fresh = importlib.reload(archive)
    folder = tmp_path / "outputs" / "run"
    folder.mkdir(parents=True)
    (folder / "config.yaml").write_text("method: fedavg\n")
    monkeypatch.setattr(fresh, "LAB", tmp_path)
    monkeypatch.setattr(fresh.time, "monotonic", lambda: 5.0)
    assert len(fresh.history()) == 1
