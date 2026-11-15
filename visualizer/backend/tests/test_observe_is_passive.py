"""The visualizer's core promise: observing a run does not change it."""
import json

import pytest
from rfl_viz.observe import observed

from rfl.experiment.config import load_config
from rfl.experiment.runner import run_experiment


def _cfg(tmp_path, name, **over):
    base = {"name": name, "method": "fedhep_plus", "model": "resnet9_multihead", "data.dataset": "synthetic",
            "data.num_clients": 6, "data.train_subset": 600, "data.test_subset": 300, "data.alpha": 0.5,
            "train.local_epochs": 1, "run.rounds": 3, "run.eval_every": 1, "run.device": "cpu", "run.seed": 3,
            "run.output_dir": str(tmp_path), "attack.kind": "sign_flip", "attack.count": 2,
            "plus.reference": "median", "plus.regroup": True}
    base.update(over)
    return load_config(overrides=base)


def _strip(result):
    out = json.loads(json.dumps(result))
    out.pop("resources")
    out.pop("name")
    for row in out["history"]:
        row.pop("elapsed_s")
    return out


@pytest.mark.parametrize("method,extra", [
    ("fedhep_plus", {}),
    ("fedhep_defended", {}),
    ("fedhep", {"attack.kind": "backdoor"}),
])
def test_observed_run_is_bit_identical_and_emits_events(tmp_path, method, extra):
    plain = run_experiment(_cfg(tmp_path, "plain", method=method, **extra))
    events = []
    with observed(events.append):
        seen = run_experiment(_cfg(tmp_path, "seen", method=method, **extra))
    assert _strip(seen) == _strip(plain)
    kinds = {e["type"] for e in events}
    assert {"round_started", "client_training_started", "client_training_completed", "round_completed"} <= kinds
    # Defended FedHEP never regroups (its groups stay random), so it must emit no grouping step at all
    assert ("grouping_step" in kinds) == (method != "fedhep_defended")
    assert sum(e["type"] == "client_training_completed" for e in events) == 6 * 3
    if method != "fedhep":
        assert any(e["type"] == "aggregation_completed" and e["tier"] == "global_defense" for e in events)
    if extra.get("attack.kind") != "backdoor":
        flips = [e for e in events if e["type"] == "attack_executed"]
        assert flips and all(e["cosine_clean_vs_sent"] < -0.99 for e in flips)   # sign flip reverses the update
