"""Defended FedHEP server rules, the trust ledger, and end-to-end runs under attack on synthetic data."""
import pytest

from rfl.defense.trust_ledger import TrustLedger
from rfl.experiment.config import load_config
from rfl.experiment.runner import run_experiment


def test_equal_share_keeps_full_trust():
    ledger = TrustLedger(4)
    for _ in range(20):
        ledger.update({c: 0.25 for c in range(4)}, 4)
    assert ledger.score == {c: pytest.approx(1.0) for c in range(4)} and not ledger.isolated


def test_zero_trust_isolates_after_nine_updates():
    """0.85^k < 0.25 first at k = 9."""
    ledger = TrustLedger(2)
    for k in range(1, 10):
        ledger.update({0: 0.0, 1: 1.0}, 2)
        assert (0 in ledger.isolated) == (k == 9)
    assert 1 not in ledger.isolated


def _cfg(tmp_path, method, kind, rate):
    return load_config(overrides={
        "name": f"smoke_{method}", "method": method, "model": "resnet9_multihead", "data.dataset": "synthetic",
        "data.num_clients": 6, "data.train_subset": 600, "data.test_subset": 300, "data.alpha": 0.5,
        "train.local_epochs": 1, "run.rounds": 3, "run.eval_every": 3, "run.device": "cpu", "run.seed": 3,
        "attack.kind": kind, "attack.rate": rate, "run.output_dir": str(tmp_path)})


@pytest.mark.parametrize("method", ["fedhep", "fedhep_defended"])
def test_runs_under_attack_and_scores_only_honest_clients(tmp_path, method):
    result = run_experiment(_cfg(tmp_path, method, "sign_flip", 0.4))
    attackers = set(result["attackers"])
    assert attackers and len(attackers) < 6  # seed 3 gives a mix
    assert set(result["final"]["per_client"]) == set(range(6)) - attackers


def test_defended_diagnostics_record_trust(tmp_path):
    result = run_experiment(_cfg(tmp_path, "fedhep_defended", "gradient_ascent", 0.4))
    last = result["diagnostics"][-1]
    assert last["round"] == 3 and set(last["trust_score"]) == set(range(6))
    honest = set(range(6)) - set(result["attackers"])
    assert set(last["global_trust"]) <= set(range(6)) - set(last["isolated"]) and honest
