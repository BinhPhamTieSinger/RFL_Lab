import json
from dataclasses import replace

import pytest
import torch
from rfl_viz.catalog import validate
from rfl_viz.observe import observed

from rfl.experiment import registry
from rfl.experiment.runner import run_experiment
from rfl.models.resnet9 import MultiHeadResNet9, ResNet9

torch.set_num_threads(2)


def stable(result):
    result = json.loads(json.dumps(result))
    result.pop("resources")
    result.pop("name")
    for row in result["history"]:
        row.pop("elapsed_s")
    return result


@pytest.mark.parametrize("method", list(registry.METHODS))
def test_all_methods_are_passively_observed(tmp_path, monkeypatch, method):
    monkeypatch.setitem(registry.MODELS, "resnet9", lambda c, n: ResNet9(c, n, base_channels=4))
    monkeypatch.setitem(registry.MODELS, "resnet9_multihead", lambda c, n: MultiHeadResNet9(c, n, base_channels=4))
    cfg = validate(
        method,
        {
            "data.dataset": "synthetic",
            "data.train_subset": 120,
            "data.test_subset": 60,
            "attack.kind": "sign_flip",
            "attack.count": 1,
            "run.rounds": 2,
        },
    )
    cfg = replace(cfg, run=replace(cfg.run, output_dir=str(tmp_path)))
    plain = run_experiment(cfg)
    events = []
    with observed(events.append):
        actual = run_experiment(replace(cfg, name="observed"))
    assert stable(actual) == stable(plain)
    assert sum(e["type"] == "client_training_completed" for e in events) == 12
    assert [e["values"]["pooled"] for e in events if e["type"] == "evaluation_completed"] == [
        r["pooled_acc"] for r in actual["history"]
    ]
    assert events[0]["type"] == "initialized"
    assert all(e["round"] > 0 for e in events if e["type"] == "client_training_started")


@pytest.mark.parametrize(
    "overrides",
    [
        {"run.rounds": 0},
        {"data.alpha": 0},
        {"attack.count": 6},
        {"train.lr": -1},
        {"run.output_dir": "/tmp/escape"},
        {"data.num_clients": 2.5},
    ],
)
def test_validation_rejects_bad_configuration(overrides):
    with pytest.raises(ValueError):
        validate("fedhep_plus", overrides)


def test_flat_methods_reject_coordinated_attack():
    with pytest.raises(ValueError):
        validate("fedavg", {"attack.kind": "alie"})


@pytest.mark.parametrize("kind", ["label_flip", "random_noise", "gradient_ascent", "backdoor", "alie", "min_max"])
def test_attack_events_match_real_execution(tmp_path, monkeypatch, kind):
    monkeypatch.setitem(registry.MODELS, "resnet9_multihead", lambda c, n: MultiHeadResNet9(c, n, base_channels=4))
    cfg = validate(
        "fedhep_plus",
        {
            "data.dataset": "synthetic",
            "data.train_subset": 120,
            "data.test_subset": 60,
            "attack.kind": kind,
            "attack.count": 2,
        },
    )
    cfg = replace(cfg, run=replace(cfg.run, output_dir=str(tmp_path)))
    events = []
    with observed(events.append):
        result = run_experiment(cfg)
    assert any(e["type"] == "attack_executed" for e in events)
    assert len(result["attackers"]) == 2
    assert any(e["type"] == "operation_completed" and e["operation"] == "backward" for e in events)


@pytest.mark.parametrize("rule", ["median", "trimmed_mean", "bulyan", "flame", "fltrust", "arc_nnm_tm"])
def test_robust_rule_integration(tmp_path, monkeypatch, rule):
    monkeypatch.setitem(registry.MODELS, "resnet9_multihead", lambda c, n: MultiHeadResNet9(c, n, base_channels=4))
    cfg = validate(
        "fedhep_robust",
        {
            "data.dataset": "synthetic",
            "data.num_clients": 7,
            "data.train_subset": 140,
            "data.test_subset": 70,
            "robust.rule": rule,
            "robust.assumed_attackers": 1,
        },
    )
    cfg = replace(cfg, run=replace(cfg.run, output_dir=str(tmp_path)))
    events = []
    with observed(events.append):
        run_experiment(cfg)
    assert any(e["type"] == "aggregation_completed" and e.get("rule") == rule for e in events)
