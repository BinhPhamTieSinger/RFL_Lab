"""IPM and the three Min-Max perturbation directions (rfl.attacks.adaptive)."""
import pytest
import torch

from rfl.attacks.adaptive import CoordinatedAttack, _within_bound, crafted_update


@pytest.fixture
def observed():
    g = torch.Generator().manual_seed(0)
    return torch.randn(6, 50, generator=g) + 0.5


def test_ipm_is_minus_epsilon_times_the_mean(observed):
    assert torch.allclose(crafted_update(observed, "ipm", 2.0), -2.0 * observed.mean(0))


def test_min_max_default_direction_is_unchanged(observed):
    a = crafted_update(observed, "min_max")
    b = crafted_update(observed, "min_max", perturbation="std")
    assert torch.equal(a, b)


@pytest.mark.parametrize("p", ["std", "unit", "sign"])
def test_min_max_stays_within_the_bound_and_uses_it(observed, p):
    out = crafted_update(observed, "min_max", perturbation=p)
    radius = _within_bound(observed)
    far = float(((out - observed) ** 2).sum(1).max())
    assert far <= radius * (1 + 1e-6)
    assert far >= radius * 0.99          # the search pushes gamma to the edge of the bound
    mean = observed.mean(0)
    direction = {"std": observed.std(0, correction=0), "unit": mean / mean.norm(), "sign": torch.sign(mean)}[p]
    gamma = float(((mean - out) @ direction) / (direction @ direction))
    assert gamma > 0
    assert torch.allclose(out, mean - gamma * direction, atol=1e-4)


def test_bad_perturbation_is_rejected():
    with pytest.raises(ValueError):
        CoordinatedAttack("min_max", perturbation="diagonal")


@pytest.mark.parametrize("over", [{"attack.kind": "ipm", "attack.ipm_epsilon": 6.0, "attack.knowledge": "omniscient"},
                                  {"attack.kind": "min_max", "attack.perturbation": "unit",
                                   "attack.knowledge": "omniscient"}])
def test_new_attacks_run_end_to_end_on_a_fedhep_server(tmp_path, over):
    from rfl.experiment.config import load_config
    from rfl.experiment.runner import run_experiment
    cfg = load_config(overrides={"name": "strength", "method": "fedhep_plus", "model": "resnet9_multihead",
                                 "data.dataset": "synthetic", "data.num_clients": 6, "data.train_subset": 600,
                                 "data.test_subset": 300, "data.alpha": 0.5, "train.local_epochs": 1,
                                 "run.rounds": 2, "run.eval_every": 2, "run.device": "cpu", "run.seed": 3,
                                 "run.output_dir": str(tmp_path), "attack.count": 2, "plus.reference": "median",
                                 **over})
    result = run_experiment(cfg)
    assert len(result["attackers"]) == 2 and result["history"]


def test_min_sum_meets_its_bound_at_the_edge(observed):
    out = crafted_update(observed, "min_sum")
    sums = ((observed[:, None, :] - observed[None, :, :]) ** 2).sum(-1).sum(1)
    mine = float(((out - observed) ** 2).sum())
    assert mine <= float(sums.min()) * (1 + 1e-6)
    assert mine >= float(sums.min()) * 0.99


def test_fang_trim_pushes_the_trimmed_mean_against_the_honest_direction():
    from rfl.attacks.adaptive import fang_trim
    honest = torch.tensor([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0], [1.5, -0.5]])
    gen = torch.Generator().manual_seed(0)
    bad = torch.stack([fang_trim(honest, True, gen) for _ in range(3)])
    assert (bad[:, 0] <= honest[:, 0].min()).all() and (bad[:, 1] >= honest[:, 1].max()).all()
    allv = torch.cat([honest, bad]).sort(0).values[1:-1]       # trim one from each end
    assert allv.mean(0)[0] < honest.mean(0)[0] and allv.mean(0)[1] > honest.mean(0)[1]
    partial = fang_trim(honest, False, gen)
    mu, sd = honest.mean(0), honest.std(0, correction=0)
    assert mu[0] - 4 * sd[0] <= partial[0] <= mu[0] - 3 * sd[0]


def test_history_attacks_wait_a_round_then_use_the_global_history():
    from rfl.attacks.adaptive import HistoryAttack
    mask = torch.ones(4, dtype=torch.bool)
    ups = {0: torch.tensor([1.0, 1.0, 1.0, 1.0]), 1: torch.tensor([2.0, 0.0, 1.0, 1.0])}
    a = HistoryAttack("poisonedfl", seed=1)
    s0 = torch.zeros(4)
    assert torch.equal(a.poison_batch(ups, s0, frozenset({1}), mask)[1], ups[1])      # round 1: no history
    s1 = torch.tensor([0.5, -0.25, 0.0, 1.0])
    out = a.poison_batch({c: u + s1 for c, u in ups.items()}, s1, frozenset({1}), mask)
    assert torch.allclose(out[1] - s1, 8.0 * s1.abs() * a.signs)
    signs = a.signs.clone()
    s2 = s1 * 2
    a.poison_batch({c: u + s2 for c, u in ups.items()}, s2, frozenset({1}), mask)
    assert torch.equal(a.signs, signs)                                                   # fixed for the run
    x = HistoryAttack("xfed")
    x.poison_batch(ups, s0, frozenset({1}), mask)
    out = x.poison_batch({c: u + s1 for c, u in ups.items()}, s1, frozenset({1}), mask)
    own = ups[1]
    assert torch.allclose(torch.nn.functional.cosine_similarity(out[1] - s1, own, dim=0).abs(), torch.tensor(1.0))


@pytest.mark.parametrize("kind", ["min_sum", "fang_trim", "poisonedfl", "xfed"])
def test_suite_attacks_run_end_to_end(tmp_path, kind):
    from rfl.experiment.config import load_config
    from rfl.experiment.runner import run_experiment
    cfg = load_config(overrides={"name": kind, "method": "fedhep_plus", "model": "resnet9_multihead",
                                 "data.dataset": "synthetic", "data.num_clients": 6, "data.train_subset": 600,
                                 "data.test_subset": 300, "data.alpha": 0.5, "train.local_epochs": 1,
                                 "run.rounds": 3, "run.eval_every": 3, "run.device": "cpu", "run.seed": 3,
                                 "run.output_dir": str(tmp_path), "attack.count": 2, "attack.kind": kind,
                                 "attack.knowledge": "omniscient", "plus.reference": "median"})
    assert run_experiment(cfg)["history"]
