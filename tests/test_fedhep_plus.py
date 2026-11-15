"""Phase 3: screening (robust z, evidence), weights-only similarity, and FedHEP+ end to end."""
import pytest
import torch
from hypothesis import example, given
from hypothesis import strategies as st

from rfl.defense.screening import EvidenceTracker, ScreenOptions, robust_z, screening_scores
from rfl.defense.soft_cosine import flat_layout
from rfl.experiment.config import load_config
from rfl.experiment.runner import run_experiment
from rfl.models.resnet9 import MultiHeadResNet9

scores_st = st.lists(st.floats(-1, 1, allow_nan=False), min_size=3, max_size=30)


@given(scores_st, st.floats(0.01, 10))
@example([0.0, 0.0, 1.0, 1.0, 1.2884714794198262e-12], 0.5)  # spread ~2e-12: fell either side of a fixed cutoff
def test_robust_z_ignores_scale(scores, a):
    """The property Defended FedHEP's fixed temperature lacks: if every cosine shrinks (a < 1) as training goes
    on, nobody's z changes."""
    s = torch.tensor(scores, dtype=torch.float64)
    z1, z2 = robust_z(s), robust_z(a * s)
    assert torch.allclose(z1, z2, atol=1e-6 * (1 + z1.abs().max().item()))


def test_robust_z_without_spread_or_with_few_senders_is_zero():
    assert torch.equal(robust_z(torch.full((5,), 0.0)), torch.zeros(5))
    assert torch.equal(robust_z(torch.tensor([0.9, 0.9, -0.9, 0.9])), torch.zeros(4))


def test_tight_consensus_does_not_magnify_small_differences():
    z = robust_z(torch.tensor([0.90, 0.91, 0.90, 0.91, 0.90, 0.86]))
    assert z.min() > -1.0  # 0.86 among ~0.90s is not an outlier


def test_evidence_isolates_persistent_outliers_only():
    t = EvidenceTracker(decay=0.7, cut=1.5)
    t.update({0: -3.0, 1: -2.6, 2: 0.5})
    assert not t.isolated                       # one bad round is not enough
    t.update({0: -3.0, 1: 0.0, 2: 0.5})
    assert t.isolated == {0}                    # two rounds at z = -3 are


def _layout():
    model = MultiHeadResNet9(3, 20, base_channels=4)
    return flat_layout(model.state_dict(), {k for k, _ in model.named_parameters()}), model


@pytest.mark.parametrize("reference", ["mean_direction", "median"])
@pytest.mark.parametrize("signal", ["root_head", "weights"])
def test_sign_flipped_updates_score_far_below_honest_ones(reference, signal):
    layout, model = _layout()
    dim = sum(v.numel() for v in model.state_dict().values())
    g = torch.Generator().manual_seed(0)
    base = torch.randn(dim, generator=g)
    ups = [base + 0.5 * torch.randn(dim, generator=g) for _ in range(12)]
    ups += [-1.5 * (base + 0.5 * torch.randn(dim, generator=g)) for _ in range(3)]
    masks = torch.ones(15, 20, dtype=torch.bool)
    z = robust_z(screening_scores(ups, layout, masks, ScreenOptions(signal, reference)))
    assert (z[12:] < -2.5).all() and (z[:12] > -2.5).all()


def test_median_reference_resists_a_coordinated_minority():
    """Five attackers pushing the same wrong direction drag the mean-of-directions reference toward them; the
    coordinate-wise median stays with the honest ten."""
    layout, model = _layout()
    dim = sum(v.numel() for v in model.state_dict().values())
    g = torch.Generator().manual_seed(1)
    honest, evil = torch.randn(dim, generator=g), torch.randn(dim, generator=g)
    ups = [honest + 1.0 * torch.randn(dim, generator=g) for _ in range(10)]
    ups += [evil + 0.1 * torch.randn(dim, generator=g) for _ in range(5)]
    gap = {}
    for ref in ("mean_direction", "median"):
        s = screening_scores(ups, layout, None, ScreenOptions("weights", ref))
        gap[ref] = (s[:10].mean() - s[10:].mean()).item()
    assert gap["median"] > gap["mean_direction"] > 0


def test_weights_only_similarity_drops_buffers():
    from rfl.client.fedhep import FedHEPTrainer
    from rfl.client.sgd import SGDSettings
    from rfl.core.types import ClientData
    from rfl.grouping.similarity import SimilarityGrouping
    from rfl.server.fedhep_server import FedHEPServer

    model = MultiHeadResNet9(3, 10)
    clients = [ClientData(torch.zeros(2, 3, 32, 32), torch.zeros(2, dtype=torch.long), torch.zeros(1, 3, 32, 32),
                          torch.zeros(1, dtype=torch.long))]
    server = FedHEPServer(model, clients, FedHEPTrainer(SGDSettings(), 1, 10, 1), SimilarityGrouping(1, 1, 0),
                          similarity_on="weights")
    n_params = sum(p.numel() for p in model.parameters())
    assert server._for_similarity({0: server.global_vec})[0].numel() == n_params


def _cfg(tmp_path, **over):
    base = {"name": "plus", "method": "fedhep_plus", "model": "resnet9_multihead", "data.dataset": "synthetic",
            "data.num_clients": 6, "data.train_subset": 600, "data.test_subset": 300, "data.alpha": 0.5,
            "train.local_epochs": 1, "run.rounds": 3, "run.eval_every": 3, "run.device": "cpu", "run.seed": 3,
            "run.output_dir": str(tmp_path), "attack.kind": "sign_flip", "attack.count": 2}
    base.update(over)
    return load_config(overrides=base)


@pytest.mark.parametrize("regroup", [False, True])
def test_fedhep_plus_leaves_out_sign_flippers(tmp_path, regroup):
    result = run_experiment(_cfg(tmp_path, **{"plus.reference": "median", "plus.regroup": regroup}))
    attackers = set(result["attackers"])
    flagged = [set(d["flagged"]) for d in result["detection"]]
    assert all(attackers <= f for f in flagged)              # both attackers out every round
    assert all(len(f - attackers) == 0 for f in flagged)    # and no honest client
    assert result["detection"][-1]["auroc"] == 1.0


def test_fedhep_weights_similarity_runs(tmp_path):
    result = run_experiment(_cfg(tmp_path, method="fedhep", **{"fedhep.similarity_on": "weights",
                                                               "attack.kind": None}))
    assert result["diagnostics"][-1]["regroup"] in ("none", "initial", "update")


def test_readmission_brings_back_a_recovered_client_and_cap_limits_exclusions():
    from rfl.defense.screening import apply_cap
    t = EvidenceTracker(decay=0.7, cut=1.5, readmit=True)
    t.update({0: -6.0})
    assert t.isolated == {0}
    for _ in range(3):
        t.update({0: 0.5})          # behaves normally again
    assert t.isolated == set()     # evidence back above -0.75
    assert apply_cap({1, 2, 3}, {1: -3.0, 2: -1.0, 3: -2.0}, {}, 2) == {1, 3}
    assert apply_cap({1}, {1: -3.0}, {}, 2) == {1}


@pytest.mark.parametrize("over", [{"plus.readmit": True}, {"plus.max_excluded": 0.34}])
def test_fedhep_plus_variants_still_exclude_sign_flippers(tmp_path, over):
    result = run_experiment(_cfg(tmp_path, **{"plus.reference": "median", **over}))
    attackers = set(result["attackers"])
    assert all(attackers <= set(d["flagged"]) for d in result["detection"])


def test_full_population_keeps_attackers_out_and_still_scores_them(tmp_path):
    result = run_experiment(_cfg(tmp_path, **{"plus.reference": "median", "plus.population": "all"}))
    attackers = {str(c) for c in result["attackers"]}
    assert all(set(result["attackers"]) <= set(d["flagged"]) for d in result["detection"])
    assert attackers <= set(map(str, result["diagnostics"][-1]["z"]))  # isolated, yet still scored



def _coordinated(seed, gamma):
    """15 honest clients of varied heterogeneity; clients 0-2 share an extra direction of strength gamma."""
    layout, model = _layout()
    dim = sum(v.numel() for v in model.state_dict().values())
    g = torch.Generator().manual_seed(seed)
    b, a = torch.randn(dim, generator=g), torch.randn(dim, generator=g)
    ups = [b + (0.5 + 2.0 * torch.rand(1, generator=g).item()) * torch.randn(dim, generator=g) for _ in range(15)]
    for i in (0, 1, 2):
        ups[i] = b + 1.2 * torch.randn(dim, generator=g) + gamma * a
    return layout, ups


def test_group_check_exposes_coordinated_attackers_that_look_normal_one_by_one():
    """The Phase 4 hypothesis on synthetic updates: three attackers sharing one direction sit inside the honest
    range individually, but the parent holding them stands out against random groups of the same size."""
    from rfl.defense.screening import group_z
    opts = ScreenOptions("weights", "median")
    groups = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]
    bad, honest, individual = [], [], []
    for seed in range(8):
        layout, ups = _coordinated(seed, 2.0)
        individual += robust_z(screening_scores(ups, layout, None, opts))[:3].tolist()
        zg = group_z(ups, layout, None, opts, groups, generator=torch.Generator().manual_seed(1)).tolist()
        bad.append(zg[0])
        honest += zg[1:]
    assert min(individual) > -2.5                      # screening alone would not catch them
    assert max(bad) < -2.4 and sum(bad) / len(bad) < -3.0
    assert sum(honest) / len(honest) > -1.5


def test_drill_down_run_reports_parent_level_detection(tmp_path):
    result = run_experiment(_cfg(tmp_path, **{"plus.reference": "median", "plus.drill_down": True,
                                              "attack.kind": "backdoor", "attack.placement": "concentrated"}))
    last = result["detection"][-1]
    assert last["malicious_groups"] == [0] and "group" in last
    assert "group_z" in result["diagnostics"][-1]


def test_member_alignment_picks_the_attackers_inside_a_flagged_parent():
    """Phase 4b: in a parent of 3 coordinated attackers + 2 honest clients, the attackers line up with the other
    members' shared deviation far more than clients outside the parent do; the honest members don't."""
    from rfl.defense.screening import member_alignment
    opts = ScreenOptions("weights", "median")
    att, hon = [], []
    for seed in range(8):
        layout, ups = _coordinated(seed, 1.0)
        z = member_alignment(ups, layout, None, opts, [0, 1, 2, 3, 4], list(range(5, 15))).tolist()
        att += z[:3]
        hon += z[3:]
    assert min(att) > 5.0 and max(hon) < 3.0


def test_member_alignment_blames_nobody_when_one_honest_outlier_makes_a_parent_odd():
    from rfl.defense.screening import member_alignment
    layout, model = _layout()
    dim = sum(v.numel() for v in model.state_dict().values())
    for seed in range(8):
        g = torch.Generator().manual_seed(seed)
        b = torch.randn(dim, generator=g)
        ups = [b + (0.5 + 2.0 * torch.rand(1, generator=g).item()) * torch.randn(dim, generator=g)
               for _ in range(15)]
        ups[0] = b + 6.0 * torch.randn(dim, generator=g)  # one very atypical honest client
        z = member_alignment(ups, layout, None, ScreenOptions("weights", "median"), [0, 1, 2, 3, 4],
                             list(range(5, 15)))
        assert z.max() < 2.5


def test_evidence_mode_drill_down_never_leaves_anyone_out_for_a_round(tmp_path):
    result = run_experiment(_cfg(tmp_path, **{"plus.reference": "median", "plus.drill_down": True,
                                              "plus.drill_mode": "evidence", "plus.min_group": 3,
                                              "fedhep.num_groups": 2,
                                              "plus.group_cut": -1e9,  # check every parent
                                              "attack.kind": "backdoor", "attack.placement": "concentrated"}))
    for d in result["diagnostics"]:
        assert set(d["left_out"]) == {c for c, v in d["z"].items() if v < -2.5}
        assert "align_z" in d
    assert any(d["align_z"] for d in result["diagnostics"])


def test_global_rule_arc_nnm_tm_runs_under_the_screen_and_differs_from_sccf(tmp_path):
    base = {"plus.reference": "median", "plus.regroup": True}
    sccf = run_experiment(_cfg(tmp_path, **base))
    l0 = run_experiment(_cfg(tmp_path, name="plus_l0", **base, **{"plus.global_rule": "arc_nnm_tm",
                                                                      "plus.global_f": 3}))
    assert set(l0["attackers"]) <= set(l0["detection"][-1]["flagged"])   # screening unchanged
    assert l0["detection"][-1]["flagged"] == sccf["detection"][-1]["flagged"]
    # with both attackers excluded, f = 3 - 2 = 1 of the 4 kept clients is still trimmed, so the step differs
    assert l0["history"][-1]["pooled_acc"] != sccf["history"][-1]["pooled_acc"]
    with pytest.raises(ValueError):
        run_experiment(_cfg(tmp_path, name="plus_bad", **{"plus.global_rule": "mean"}))
