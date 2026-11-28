"""Mixed attacker populations (rfl.attacks.mixed): subgroups, staggered starts, per-subgroup coordination."""
import pytest
import torch

from rfl.attacks.adaptive import CoordinatedAttack
from rfl.attacks.backdoor import BackdoorAttack
from rfl.attacks.byzantine import ByzantineAttack
from rfl.attacks.mixed import MixedAttack


def test_attackers_are_dealt_to_parts_in_order_and_parts_start_late():
    sign, bd = ByzantineAttack("sign_flip"), BackdoorAttack()
    m = MixedAttack([(sign, 1, 3), (bd, 2, 0)], frozenset({7, 2, 5}))
    assert [g[1] for g in m.groups] == [frozenset({2}), frozenset({5, 7})]
    assert m.part(2, 2) is None and m.part(2, 3) is sign
    assert m.part(5, 1) is bd and m.part(0, 1) is None
    assert m.trigger == bd.trigger


def test_counts_must_match_the_attackers():
    with pytest.raises(ValueError):
        MixedAttack([(ByzantineAttack("sign_flip"), 2, 0)], frozenset({1}))


def test_coordinated_part_uses_only_its_own_subgroup():
    alie = CoordinatedAttack("alie", 1.0)
    m = MixedAttack([(alie, 2, 0), (ByzantineAttack("sign_flip"), 1, 0)], frozenset({0, 1, 2}))
    start = torch.zeros(4)
    uploads = {0: torch.ones(4), 1: 3 * torch.ones(4), 2: 100 * torch.ones(4), 3: torch.zeros(4)}
    out = m.poison_batch(uploads, start, 1, torch.ones(4, dtype=torch.bool))
    assert torch.allclose(out[0], torch.full((4,), 1.0)) and torch.equal(out[0], out[1])   # mean 2 - 1 * std 1
    assert torch.equal(out[2], uploads[2])


def _cfg(tmp_path, mix):
    from rfl.experiment.config import load_config
    return load_config(overrides={"name": "mix", "method": "fedhep_plus", "model": "resnet9_multihead",
                                  "data.dataset": "synthetic", "data.num_clients": 6, "data.train_subset": 600,
                                  "data.test_subset": 300, "data.alpha": 0.5, "train.local_epochs": 1,
                                  "run.rounds": 3, "run.eval_every": 3, "run.device": "cpu", "run.seed": 3,
                                  "run.output_dir": str(tmp_path), "attack.kind": "mixed", "attack.mix": mix,
                                  "plus.reference": "median", "plus.signal": "weights_critical"})


def test_mixed_run_end_to_end_measures_the_backdoor(tmp_path):
    from rfl.experiment.runner import run_experiment
    result = run_experiment(_cfg(tmp_path, [{"kind": "stealth_backdoor", "count": 1},
                                            {"kind": "sign_flip", "count": 1, "start_round": 2}]))
    assert len(result["attackers"]) == 2 and result["history"][-1]["asr"] is not None


def test_mixed_refuses_a_count(tmp_path):
    from rfl.experiment.config import load_config
    from rfl.experiment.registry import _attack
    cfg = load_config(overrides={"attack.kind": "mixed", "attack.count": 2, "method": "fedhep_plus",
                                 "attack.mix": [{"kind": "sign_flip", "count": 2}]})
    with pytest.raises(ValueError):
        _attack(cfg, 6)
