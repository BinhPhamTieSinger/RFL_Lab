"""Sentence-based readmission (EvidenceTracker) and attack schedules (sleeper, on-off)."""
import pytest

from rfl.defense.screening import EvidenceTracker, ScreenOptions


def _round(t: EvidenceTracker, z: dict) -> set:
    t.release()
    t.update({c: v for c, v in z.items() if c not in t.isolated})
    return set(t.isolated)


def test_without_a_sentence_isolation_is_permanent():
    t = EvidenceTracker(0.7, 1.5, patience=3)
    seen = [_round(t, {0: -5.0, 1: 0.0}) for _ in range(3)] + [_round(t, {0: 0.0, 1: 0.0}) for _ in range(20)]
    assert seen[1] == set() and seen[2] == {0} and all(s == {0} for s in seen[2:])


def test_sentence_returns_the_client_resets_evidence_and_the_second_strike_is_permanent():
    t = EvidenceTracker(0.7, 1.5, patience=3, sentence=5, max_strikes=2, probation_patience=1)
    for _ in range(3):
        _round(t, {0: -5.0})
    assert t.isolated == {0} and t.strikes[0] == 1
    for _ in range(4):                      # rounds 4-7: serving the sentence
        assert _round(t, {0: 0.0}) == {0}
    assert _round(t, {0: 0.0}) == set()      # round 8: back, scored again, evidence reset
    assert t.evidence[0] == 0.0 and 0 in t.returned
    assert _round(t, {0: -9.0}) == {0}       # probation: one bad round is enough
    assert t.strikes[0] == 2
    for _ in range(20):
        assert _round(t, {0: 0.0}) == {0}    # second strike: permanent


def test_honest_client_that_comes_back_clean_stays():
    t = EvidenceTracker(0.7, 1.5, patience=3, sentence=2, max_strikes=2, probation_patience=1)
    for _ in range(3):
        _round(t, {0: -5.0})
    seen = [_round(t, {0: 0.3}) for _ in range(10)]
    assert seen[-1] == set() and t.strikes[0] == 1


def test_sentence_and_readmit_are_exclusive():
    with pytest.raises(ValueError):
        ScreenOptions("weights", readmit=True, sentence=5)


def test_attack_schedule_sleeper_and_on_off():
    from rfl.server.fedhep_server import FedHEPServer
    s = FedHEPServer.__new__(FedHEPServer)
    s.attack_stop, s.attack_start, s.attack_on, s.attack_off = 0, 0, 0, 0
    assert all(s.attacking(r) for r in range(1, 31))
    s.attack_start = 10
    assert [s.attacking(r) for r in (9, 10, 30)] == [False, True, True]
    s.attack_on, s.attack_off = 2, 3
    assert [r for r in range(10, 21) if s.attacking(r)] == [10, 11, 15, 16, 20]
    s.attack_stop = 15
    assert [r for r in range(10, 21) if s.attacking(r)] == [10, 11, 15]


@pytest.mark.parametrize("over", [{"plus.sentence": 2, "plus.probation_patience": 1},
                                  {"attack.kind": "backdoor", "attack.start_round": 2}])
def test_sentence_and_sleeper_run_end_to_end(tmp_path, over):
    from rfl.experiment.config import load_config
    from rfl.experiment.runner import run_experiment
    cfg = load_config(overrides={"name": "readmit", "method": "fedhep_plus", "model": "resnet9_multihead",
                                 "data.dataset": "synthetic", "data.num_clients": 6, "data.train_subset": 600,
                                 "data.test_subset": 300, "data.alpha": 0.5, "train.local_epochs": 1,
                                 "run.rounds": 5, "run.eval_every": 5, "run.device": "cpu", "run.seed": 3,
                                 "run.output_dir": str(tmp_path), "attack.kind": "sign_flip", "attack.count": 2,
                                 "plus.reference": "median", **over})
    result = run_experiment(cfg)
    assert result["history"]


def test_sentence_with_rewind_is_refused(tmp_path):
    from rfl.experiment.config import load_config
    from rfl.experiment.runner import run_experiment
    cfg = load_config(overrides={"name": "x", "method": "fedhep_plus", "model": "resnet9_multihead",
                                 "data.dataset": "synthetic", "data.num_clients": 6, "data.train_subset": 600,
                                 "data.test_subset": 300, "run.rounds": 1, "run.device": "cpu",
                                 "run.output_dir": str(tmp_path), "plus.sentence": 3, "plus.rewind": "trunk_root"})
    with pytest.raises(ValueError):
        run_experiment(cfg)
