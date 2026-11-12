"""Server behaviour and an end-to-end FedAvg run on synthetic data."""
import json

import pytest
import torch

from rfl.aggregation.fedavg import FedAvg
from rfl.client.sgd import LocalRule
from rfl.core.types import ClientData
from rfl.experiment.config import load_config
from rfl.experiment.runner import run_experiment
from rfl.server.fedavg_server import ConsensusServer


class ShiftTrainer(LocalRule):
    """Adds (client id + 1) to every floating-point entry, so the expected global state is known exactly."""

    def __init__(self):
        pass

    def train(self, model, data, round_idx, client_id=0, targets=None):
        with torch.no_grad():
            for v in model.state_dict().values():
                if v.is_floating_point():
                    v.add_(float(data.y_train[0]) + 1.0)


def test_server_moves_global_to_sample_weighted_mean():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    sizes = [10, 30, 60]
    clients = [ClientData(torch.zeros(n, 3), torch.full((n,), i), torch.zeros(1, 3), torch.zeros(1))
               for i, n in enumerate(sizes)]
    ConsensusServer(model, clients, ShiftTrainer(), FedAvg()).run_round(1)
    expected_shift = sum(n * (i + 1) for i, n in enumerate(sizes)) / sum(sizes)
    for k, v in model.state_dict().items():
        if v.is_floating_point():  # includes BatchNorm running statistics, as in the authors' code
            assert torch.allclose(v, before[k] + expected_shift), k


def test_unknown_setting_is_rejected():
    with pytest.raises(ValueError, match="Unknown setting"):
        load_config(overrides={"train.learning_rate": 0.1})


def test_text_in_a_numeric_setting_is_rejected():
    """PyYAML reads -1.0e9 (no sign in the exponent) as text; that once crashed a sweep 12 runs in."""
    import yaml
    with pytest.raises(ValueError, match="must be a number"):
        load_config(overrides=yaml.safe_load("{plus.group_cut: -1.0e9}"))
    assert load_config(overrides=yaml.safe_load("{plus.group_cut: -1.0e+9}")).plus.group_cut == -1e9


def test_every_sweep_config_loads():
    from pathlib import Path

    import yaml

    from rfl.experiment.sweep import expand
    for spec in sorted(Path("configs/sweeps").glob("*.yaml")):
        assert expand(yaml.safe_load(spec.read_text())), spec


def test_fedavg_end_to_end_learns_and_writes_results(tmp_path):
    cfg = load_config(overrides={
        "name": "smoke", "data.dataset": "synthetic", "data.num_clients": 4,
        "data.train_subset": 400, "data.test_subset": 200, "data.alpha": 0.5,
        "train.local_epochs": 1, "run.rounds": 3, "run.eval_every": 1, "run.device": "cpu",
        "run.output_dir": str(tmp_path)})
    result = run_experiment(cfg)
    assert [h["round"] for h in result["history"]] == [1, 2, 3]
    assert result["final"]["pooled"] > 25.0  # 10 classes: chance is 10%
    saved = json.loads((tmp_path / "smoke_s42" / "metrics.json").read_text())
    assert saved["final"]["pooled"] == result["final"]["pooled"]
    assert (tmp_path / "smoke_s42" / "config.yaml").exists()
