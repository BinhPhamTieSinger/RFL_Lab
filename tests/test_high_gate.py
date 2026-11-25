"""plus.high_gate: with a two-sided screen, the high side (clients unusually close to the others) acts only in
rounds whose gap ratio reaches high_gate; the low side is unchanged (report 8.2)."""
from rfl.experiment.runner import run_experiment
from tests.test_fedhep_plus import _cfg

UNION = {"plus.reference": "median", "plus.signal": "weights_critical", "plus.two_sided": True,
         "plus.global_rule": "arc_nnm_tm", "plus.global_f": 2, "plus.gap_gate": 0.0}


def test_unreachable_high_gate_leaves_out_only_the_low_side(tmp_path):
    result = run_experiment(_cfg(tmp_path, name="hg", **{**UNION, "plus.high_gate": 1e9}))
    left = [(d, c) for d in result["diagnostics"] for c in d["left_out"]]
    assert left                                                  # the sign flippers are left out ...
    assert all(d["z"][c] < 0 for d, c in left)              # ... and only on the low side


def test_default_high_gate_keeps_the_previous_screen(tmp_path):
    a = run_experiment(_cfg(tmp_path / "a", name="a", **UNION))
    b = run_experiment(_cfg(tmp_path / "b", name="b", **{**UNION, "plus.high_gate": 0.0}))
    assert [d["left_out"] for d in a["diagnostics"]] == [d["left_out"] for d in b["diagnostics"]]
