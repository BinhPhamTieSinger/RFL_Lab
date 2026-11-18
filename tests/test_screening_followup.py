"""Regression checks for scale floors, persistence and coordinated attack constraints."""
import pytest
import torch

from rfl.attacks.adaptive import CoordinatedAttack, StealthBackdoorAttack, crafted_update, stealth_blend
from rfl.core.types import DetectionResult, RoundReport
from rfl.defense.screening import EvidenceTracker, one_sided, robust_z
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


def test_two_sided_screening_acts_on_clients_far_above_the_others():
    z = {0: 5.0, 1: -3.0, 2: 0.4}
    assert one_sided(z, two_sided=False) == z
    assert one_sided(z, two_sided=True) == {0: -5.0, 1: -3.0, 2: -0.4}


def test_gap_ratio_is_small_for_a_spread_and_large_for_a_split():
    from rfl.defense.screening import gap_ratio
    assert gap_ratio([0.1, 0.2, 0.3]) == 0.0                          # too few to judge
    spread = [i / 10 for i in range(12)]                              # evenly spread: ratio 1
    assert abs(gap_ratio(spread) - 1.0) < 1e-9
    split = spread + [-5.0, -5.1, -5.05]                              # three far below: a clear split
    assert gap_ratio(split) > 10
    assert gap_ratio(spread + [9.0]) > 10                             # a split above counts too


def test_stealth_blend_keeps_an_inside_update_and_pulls_a_far_one_to_the_bound():
    honest = torch.randn(8, 19, generator=torch.Generator().manual_seed(7))
    radius = torch.cdist(honest, honest).square().max()
    inside = honest.mean(0) + 0.1 * (honest[0] - honest.mean(0))
    out, lam = stealth_blend(inside, honest)
    assert lam == 1.0 and torch.equal(out, inside)
    far = honest.mean(0) + 50 * torch.ones(19)
    out, lam = stealth_blend(far, honest)
    assert 0 < lam < 1
    assert (out - honest).square().sum(1).max() <= radius + 1e-4
    further = honest.mean(0) + 1.01 * (out - honest.mean(0))
    assert (further - honest).square().sum(1).max() > radius


def test_stealth_backdoor_changes_only_attackers_parameters():
    attack = StealthBackdoorAttack()
    start = torch.zeros(3)
    uploads = {0: torch.tensor([90., 90., 10.]), 1: torch.tensor([2., 3., 20.]),
               2: torch.tensor([3., 1., 30.]), 3: torch.tensor([1., 2., 40.])}
    out = attack.poison_batch(uploads, start, frozenset({0}), torch.tensor([True, True, False]))
    assert out[0][2] == 10                      # buffer kept
    assert out[0][:2].abs().max() < 90          # pulled in
    assert attack.blend and 0 < attack.blend[0] < 1
    assert all(torch.equal(out[c], uploads[c]) for c in (1, 2, 3))

