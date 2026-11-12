"""Regression checks for scale floors, persistence and coordinated attack constraints."""
import pytest
import torch

from rfl.attacks.adaptive import CoordinatedAttack, crafted_update
from rfl.core.types import DetectionResult, RoundReport
from rfl.defense.screening import EvidenceTracker, robust_z
from rfl.evaluation.detection import detection_metrics


def test_absolute_floor_bounds_z_near_zero_median():
    scores = torch.tensor([-0.02, 0., 0., 0., 0., 0.0001])
    assert robust_z(scores, absolute_floor=0.02).min() == -1


def test_patience_requires_consecutive_bad_rounds():
    tracker = EvidenceTracker(0.7, 1.5, patience=3)
    for z in [-10., -10., 0., -10., -10.]:
        tracker.update({0: z})
        assert not tracker.isolated
    tracker.update({0: -10.})
    assert tracker.isolated == {0}


def test_false_exclusion_uses_all_honest_clients_as_denominator():
    report = RoundReport(DetectionResult({}, frozenset({0, 1, 2})))
    metrics = detection_metrics(report, {0, 1}, 6)
    assert metrics['precision'] == 2 / 3
    assert metrics['recall'] == 1
    assert metrics['false_exclusion_rate'] == 1 / 4


def test_min_max_satisfies_distance_constraint_and_is_boundary_solution():
    observed = torch.randn(8, 19, generator=torch.Generator().manual_seed(42))
    crafted = crafted_update(observed, 'min_max')
    radius = torch.cdist(observed, observed).square().max()
    assert (crafted - observed).square().sum(1).max() <= radius + 1e-4
    mean = observed.mean(0)
    further = mean + 1.01 * (crafted - mean)
    assert (further - observed).square().sum(1).max() > radius


@pytest.mark.parametrize('kind', ['alie', 'min_max'])
def test_coordinated_attack_preserves_buffers_and_honest_uploads(kind):
    attack = CoordinatedAttack(kind, knowledge='omniscient')
    start = torch.zeros(3)
    uploads = {0: torch.tensor([1., 2., 10.]), 1: torch.tensor([2., 3., 20.]),
               2: torch.tensor([3., 1., 30.])}
    out = attack.poison_batch(uploads, start, frozenset({0}), torch.tensor([True, True, False]))
    assert out[0][2] == 10
    assert torch.equal(out[1], uploads[1])
    assert torch.equal(out[2], uploads[2])
    assert torch.equal(attack.poison_labels(torch.arange(3), 3), torch.arange(3))
