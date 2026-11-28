"""Fixes for failures found on the Acer (report 8.2): windowed patience, permanent two-signal isolations under a
sentence, rewind only for two-signal isolations, and a floor on the global step's assumed attacker count."""
import pytest

from rfl.defense.screening import EvidenceTracker, ScreenOptions


def _round(t: EvidenceTracker, z: dict, permanent=frozenset()) -> set:
    t.release()
    t.update({c: v for c, v in z.items() if c not in t.isolated}, permanent)
    return set(t.isolated)


ON_OFF = [-5.0, -5.0, 0.0] * 4          # 2 bad rounds, 1 clean, repeated


def test_two_on_one_off_is_never_isolated_in_a_row_but_is_within_a_window():
    streak = EvidenceTracker(0.7, 1.5, patience=3)
    assert all(_round(streak, {0: v}) == set() for v in ON_OFF)
    window = EvidenceTracker(0.7, 1.5, patience=3, window=5)
    seen = [_round(window, {0: v}) for v in ON_OFF]
    assert seen[2] == set() and seen[3] == {0}      # rounds 1, 2, 4 bad: the third bad round within 5 isolates


def test_window_still_needs_low_evidence_and_forgets_old_rounds():
    t = EvidenceTracker(0.7, 1.5, patience=3, window=5)
    for v in (-5.0, -5.0, 0.0, 0.0, 0.0, 0.0, -5.0, 0.0, 0.0):  # never 3 bad within any 5
        assert _round(t, {0: v}) == set()


def test_window_must_cover_the_patience():
    with pytest.raises(ValueError):
        ScreenOptions("weights", isolation_patience=3, isolation_window=2)


def test_permanent_isolation_skips_the_sentence():
    t = EvidenceTracker(0.7, 1.5, patience=3, sentence=2, max_strikes=2)
    for _ in range(3):
        _round(t, {0: -5.0, 1: -5.0}, permanent={1})
    assert t.isolated == {0, 1} and t.strikes == {0: 1, 1: 2}
    for _ in range(5):
        seen = _round(t, {0: 0.0})
    assert seen == {1}                                 # client 0 served its sentence, client 1 never comes back


def test_sentence_single_needs_a_sentence():
    with pytest.raises(ValueError):
        ScreenOptions("weights_critical", sentence_single=True)
    ScreenOptions("weights_critical", sentence=5, sentence_single=True)


def _cfg(tmp_path, **over):
    from rfl.experiment.config import load_config
    return load_config(overrides={"name": "fix", "method": "fedhep_plus", "model": "resnet9_multihead",
                                  "data.dataset": "synthetic", "data.num_clients": 6, "data.train_subset": 600,
                                  "data.test_subset": 300, "data.alpha": 0.5, "train.local_epochs": 1,
                                  "run.rounds": 4, "run.eval_every": 4, "run.device": "cpu", "run.seed": 3,
                                  "run.output_dir": str(tmp_path), "attack.kind": "sign_flip", "attack.count": 2,
                                  "plus.reference": "median", "plus.signal": "weights_critical",
                                  "plus.two_sided": True, "plus.gap_gate": 8, "plus.isolation_patience": 3,
                                  "plus.global_rule": "arc_nnm_tm", "plus.global_f": 2, **over})


@pytest.mark.parametrize("over", [{"plus.isolation_window": 5},
                                  {"plus.sentence": 2, "plus.sentence_single": True},
                                  {"plus.rewind": "trunk_root", "plus.rewind_agree": True},
                                  {"plus.global_f_floor": 0.5}])
def test_fixes_run_end_to_end(tmp_path, over):
    from rfl.experiment.runner import run_experiment
    result = run_experiment(_cfg(tmp_path, **over))
    assert result["history"] and "two_signal" in result["diagnostics"][-1]


@pytest.mark.parametrize("over", [{"plus.rewind_agree": True},
                                  {"plus.sentence": 2, "plus.sentence_single": True, "plus.signal": "weights"}])
def test_two_signal_options_need_a_union_signal_and_a_rewind(tmp_path, over):
    from rfl.experiment.runner import run_experiment
    with pytest.raises(ValueError):
        run_experiment(_cfg(tmp_path, **over))


def test_rewind_agree_skips_single_signal_isolations():
    """A newly isolated client that was never flagged by both signals does not trigger the rewind."""
    from rfl.server.fedhep_plus_server import FedHEPPlusServer
    s = FedHEPPlusServer.__new__(FedHEPPlusServer)
    s.both_recent = {0: [False, False, True], 1: [False] * 5, 2: [True, False, False, False, False]}
    assert s.two_signal() == {0, 2}

