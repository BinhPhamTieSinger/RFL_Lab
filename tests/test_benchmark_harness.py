"""Phase 2 testbench: detection metrics, attacker choice and placement, the backdoor attack and its success rate,
and end-to-end runs that record them."""
import dataclasses

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from rfl.attacks.backdoor import BackdoorAttack, stamp_trigger
from rfl.attacks.byzantine import select_attackers_fixed
from rfl.core.types import ClientData, DetectionResult, RoundReport
from rfl.evaluation.backdoor import attack_success_rate
from rfl.evaluation.detection import auroc, detection_metrics, malicious_groups, prf
from rfl.experiment.config import load_config
from rfl.experiment.runner import run_experiment
from rfl.grouping.similarity import concentrate, random_assignment, spread

# ----------------------------------------------------------------------------- detection metrics


def test_prf_cases():
    assert prf({1, 2}, {2, 3}) == {"precision": 0.5, "recall": 0.5, "f1": 0.5}
    assert prf(set(), {1}) == {"precision": None, "recall": 0.0, "f1": 0.0}  # flagged nothing, missed everyone
    assert prf({1}, set()) == {"precision": 0.0, "recall": None, "f1": None}  # no attackers: recall undefined
    assert prf({1, 2}, {1, 2}) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_auroc_cases():
    assert auroc({0: 1.0, 1: 0.9, 2: 0.1}, {2}) == 1.0     # attacker trusted least
    assert auroc({0: 0.1, 1: 0.2, 2: 0.9}, {2}) == 0.0     # attacker trusted most
    assert auroc({0: 0.5, 1: 0.5}, {1}) == 0.5             # tie
    assert auroc({0: 0.5, 1: 0.5}, set()) is None          # one class only


def test_parent_is_malicious_only_with_attacker_majority():
    groups = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
    assert malicious_groups(groups, {0, 1, 3}) == {0}  # 2 of 3 in parent 0; 1 of 2 in parent 1 is no majority
    assert malicious_groups(groups, {3, 4, 5}) == {1}


def test_detection_metrics_with_parent_check():
    report = RoundReport(DetectionResult({0: 1.0, 1: 0.2, 2: 0.1}, frozenset({2})), groups={0: 0, 1: 1, 2: 1},
                         flagged_groups=frozenset({1}))
    m = detection_metrics(report, {1, 2})
    assert m["recall"] == 0.5 and m["precision"] == 1.0 and m["auroc"] == 1.0
    assert m["malicious_groups"] == [1] and m["group"]["f1"] == 1.0


# ----------------------------------------------------------------------------- attacker choice and placement


@given(st.integers(0, 15), st.integers(0, 10_000))
def test_fixed_count_is_exact_and_repeatable(count, seed):
    chosen = select_attackers_fixed(15, count, seed)
    assert len(chosen) == count and chosen == select_attackers_fixed(15, count, seed)


@given(st.integers(0, 10_000), st.integers(2, 5), st.sets(st.integers(0, 14), max_size=8))
def test_concentrate_puts_attackers_in_one_parent(seed, k, chosen):
    before = random_assignment(15, k, seed)
    after = concentrate(before, chosen, k)
    cap = max(len(chosen), -(-15 // k))
    assert all(after[c] == 0 for c in chosen)
    assert sum(g == 0 for g in after.values()) <= cap
    assert all(after[c] == before[c] for c in after if before[c] != 0 and c not in chosen)  # others stay put


# ----------------------------------------------------------------------------- backdoor


def test_trigger_changes_only_the_corner_patch():
    x = torch.zeros(2, 3, 32, 32)
    out = stamp_trigger(x)
    assert (out[:, :, :-3, :] == 0).all() and (out[:, :, :, :-3] == 0).all() and (out[:, :, -3:, -3:] > 0).all()
    assert (x == 0).all()  # input untouched


def test_backdoor_poisons_a_fixed_fraction_once():
    data = ClientData(torch.zeros(40, 3, 32, 32), torch.full((40,), 7), torch.zeros(1, 3, 32, 32), torch.zeros(1))
    attack = BackdoorAttack(target=2, fraction=0.25, seed=1)
    poisoned = attack.poison_data(5, data)
    assert int((poisoned.y_train == 2).sum()) == 10
    assert attack.poison_data(5, data) is poisoned  # same poisoned set every round
    start, trained = torch.zeros(4), torch.ones(4)
    assert torch.equal(BackdoorAttack(scale=5.0).poison_upload(trained, start), torch.full((4,), 5.0))


def test_attack_success_rate_counts_non_target_images_of_scored_clients():
    clients = [ClientData(torch.zeros(1), torch.zeros(1), torch.zeros(4, 3, 32, 32), torch.tensor([0, 1, 1, 2]))
               for _ in range(2)]
    always_target = lambda cid, x: torch.nn.functional.one_hot(torch.zeros(len(x), dtype=torch.long), 3).float()  # noqa: E731
    never = lambda cid, x: torch.nn.functional.one_hot(torch.ones(len(x), dtype=torch.long), 3).float()  # noqa: E731
    assert attack_success_rate(always_target, clients, stamp_trigger, 0) == 100.0
    assert attack_success_rate(never, clients, stamp_trigger, 0) == 0.0
    assert attack_success_rate(always_target, clients, stamp_trigger, 0, exclude=frozenset({0, 1})) is None


# ----------------------------------------------------------------------------- end to end


def _cfg(tmp_path, method, **attack):
    over = {"name": f"h_{method}", "method": method, "data.dataset": "synthetic", "data.num_clients": 6,
            "data.train_subset": 600, "data.test_subset": 300, "data.alpha": 0.5, "train.local_epochs": 1,
            "run.rounds": 2, "run.eval_every": 2, "run.device": "cpu", "run.output_dir": str(tmp_path),
            "model": "resnet9_multihead" if method.startswith("fedhep") else "resnet9"}
    over.update({f"attack.{k}": v for k, v in attack.items()})
    return load_config(overrides=over)


@pytest.mark.parametrize("method", ["fedavg", "multikrum", "fedhep", "fedhep_defended"])
def test_backdoor_run_records_asr_and_detection_every_round(tmp_path, method):
    result = run_experiment(_cfg(tmp_path, method, kind="backdoor", count=2))
    assert len(result["attackers"]) == 2
    assert result["history"][-1]["asr"] is not None
    assert [d["round"] for d in result["detection"]] == [1, 2]
    heads = result["history"][-1].get("asr_heads")
    if method.startswith("fedhep"):
        assert set(heads) == {"root", "parent", "local"} and all(0 <= v <= 100 for v in heads.values())
    else:
        assert heads is None
    if method == "multikrum":
        assert len(result["detection"][-1]["flagged"]) == 6 - 3  # keeps m = 3
        assert result["detection"][-1]["auroc"] is not None


def test_concentrated_placement_makes_parent_zero_malicious(tmp_path):
    result = run_experiment(_cfg(tmp_path, "fedhep_defended", kind="sign_flip", count=2, placement="concentrated"))
    assert result["detection"][0]["malicious_groups"] == [0]


def test_sweep_runs_every_combination_once_and_summary_aggregates_seeds(tmp_path):
    from rfl.experiment.summary import summarize, to_markdown
    from rfl.experiment.sweep import expand, run_sweep

    spec = {"seeds": [1, 2],
            "common": {"data.dataset": "synthetic", "data.num_clients": 4, "data.train_subset": 200,
                       "data.test_subset": 100, "train.local_epochs": 1, "run.rounds": 1, "run.device": "cpu",
                       "run.output_dir": str(tmp_path)},
            "axes": [{"avg": {"method": "fedavg"}, "krum": {"method": "multikrum"}},
                     {"clean": {}, "sign": {"attack.kind": "sign_flip", "attack.count": 1}}]}
    assert [n for n, _ in expand(spec)] == ["avg__clean", "avg__sign", "krum__clean", "krum__sign"]
    assert len(run_sweep(spec)) == 8
    assert run_sweep(spec) == []  # everything already done
    rows = {r["name"]: r for r in summarize(tmp_path)}
    assert rows["krum__sign"]["seeds"] == [1, 2] and rows["krum__sign"]["attackers"] == [1, 1]
    assert rows["krum__sign"]["auroc"] is not None and rows["avg__clean"]["recall"] is None
    assert "krum__sign" in to_markdown(list(rows.values()))


def test_sweep_shards_split_the_grid_without_overlap_and_extra_overrides_every_run(tmp_path):
    from rfl.experiment.sweep import expand, run_sweep

    spec = {"seeds": [1, 2, 3],
            "common": {"data.alpha": 0.5},
            "axes": [{"a": {"method": "fedavg"}, "b": {"method": "multikrum", "data.alpha": 0.1}}]}
    extra = {"run.output_dir": str(tmp_path), "data.alpha": 0.3}
    assert {c.data.alpha for _, c in expand(spec, extra)} == {0.3}
    shards = [run_sweep(spec, dry_run=True, extra=extra, shard=(i, 4)) for i in range(4)]
    every = [f for s in shards for f in s]
    assert sorted(every) == sorted(run_sweep(spec, dry_run=True, extra=extra)) and len(set(every)) == 6


@given(st.integers(0, 10_000), st.integers(2, 5), st.sets(st.integers(0, 14), max_size=8))
def test_spread_caps_attackers_per_parent(seed, k, chosen):
    after = spread(random_assignment(15, k, seed), chosen, k)
    per_group = [sum(after[c] == g for c in chosen) for g in range(k)]
    assert max(per_group) <= -(-len(chosen) // k) if chosen else True


def test_stop_round_switches_attackers_to_honest_training(tmp_path, monkeypatch):
    """Label-flip attackers train on flipped labels up to stop_round and on their true labels after it; methods
    outside the FedHEP family refuse the option."""
    from rfl.client.fedhep import FedHEPTrainer

    poisoned: list[int] = []
    train = FedHEPTrainer.train

    def spy(self, model, data, round_idx, mem, targets=None):
        if targets is not None:
            poisoned.append(round_idx)
        return train(self, model, data, round_idx, mem, targets)

    monkeypatch.setattr(FedHEPTrainer, "train", spy)
    cfg = _cfg(tmp_path, "fedhep", kind="label_flip", count=2, stop_round=1)
    run_experiment(dataclasses.replace(cfg, run=dataclasses.replace(cfg.run, rounds=3, eval_every=3)))
    assert poisoned == [1, 1]  # two attackers, round 1 only
    with pytest.raises(ValueError):
        run_experiment(_cfg(tmp_path / "avg", "fedavg", kind="sign_flip", count=2, stop_round=1))


def test_exit_guard_leaves_training_and_main_metrics_unchanged(tmp_path):
    """The guard is evaluation-only: with it on, every unguarded metric equals the run without it, and the guarded
    metrics are recorded next to them."""
    import dataclasses as dc
    base = _cfg(tmp_path / "off", "fedhep", kind="backdoor", count=2)
    off = run_experiment(base)
    on_run = dc.replace(base.run, output_dir=str(tmp_path / "on"))
    guard_on = dc.replace(base.fedhep, exit_guard=True)
    on = run_experiment(dc.replace(base, name="guard_on", run=on_run, fedhep=guard_on))
    for a, b in zip(off["history"], on["history"], strict=True):
        assert {k: v for k, v in a.items() if k != "elapsed_s"} == \
            {k: v for k, v in b.items() if k not in ("elapsed_s", "guarded")}
    g = on["history"][-1]["guarded"]
    assert 0 <= g["pooled_acc"] <= 100 and 0 <= g["clean_flag_rate"] <= 1 and g["asr"] is not None
    with pytest.raises(ValueError):
        avg = _cfg(tmp_path / "avg", "fedavg")
        run_experiment(dc.replace(avg, fedhep=dc.replace(base.fedhep, exit_guard=True)))


def test_exit_guard_refuses_a_trigger_the_trunk_learned():
    """Unit check of the decision rule: inputs whose final class gets little support from the exits are re-decided
    by the exits; inputs the exits agree with keep the final prediction."""
    from types import SimpleNamespace

    from rfl.defense.exit_guard import ExitGuard, _Exit
    C = 4
    guard = ExitGuard(num_classes=C)
    eye = torch.eye(C)
    exit_ = _Exit(torch.zeros(C), torch.ones(C), 10 * eye, torch.zeros(C))   # exit predicts the feature's argmax
    guard.exits[0] = [exit_, exit_, exit_]
    guard.tau[0] = 0.5
    final_logits = torch.tensor([[0.0, 9.0, 0.0, 0.0], [0.0, 9.0, 0.0, 0.0]])   # final model says class 1 for both
    feats = [torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])] * 3        # exits: class 1, then class 2
    server = SimpleNamespace(predict=lambda cid, x: final_logits, memory=[SimpleNamespace(active_mask=None)])
    guard._features = lambda server, cid, images: feats
    out = guard.predict(server, 0, torch.zeros(2, 3, 32, 32)).argmax(dim=1).tolist()
    assert out == [1, 2]
    assert guard.flagged == 1 and guard.seen == 2
