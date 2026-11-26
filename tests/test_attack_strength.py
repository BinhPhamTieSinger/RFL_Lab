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
