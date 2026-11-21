"""Phase 7: the peer reference (each client scored against the median of the clients holding the most similar
classes) and shadow scoring (a second reference logged without acting)."""
import math

import pytest
import torch

from rfl.defense.screening import (
    ScreenOptions,
    critical_layer_scores,
    peer_references,
    peer_sets,
    robust_z,
    screening_scores,
)
from rfl.experiment.runner import run_experiment
from tests.test_fedhep_plus import _cfg, _layered, _layout


def _masks(rows):
    return torch.tensor(rows, dtype=torch.bool)


def test_peer_sets_pick_the_most_similar_class_sets_without_the_client_itself():
    m = _masks([[1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 1, 1], [0, 0, 1, 1]])
    peers = peer_sets(m, share=0.25)                       # ceil(0.25 x 4) = 1 peer each
    assert [p.tolist() for p in peers] == [[1], [0], [0], [4], [3]]
    for i, p in enumerate(peer_sets(m, share=2 / 3)):
        assert i not in p.tolist() and len(p) == math.ceil(2 / 3 * 4)
    assert len(peer_sets(m, share=1 / 3)[0]) == 2


def test_peer_sets_use_reference_rows_only_and_break_ties_by_index():
    m = torch.ones(5, 3, dtype=torch.bool)                 # all equally similar: ties everywhere
    rows = torch.tensor([True, False, True, True, True])   # client 1 is not in the reference
    peers = peer_sets(m, share=0.5, reference_rows=rows)
    assert peers[1].tolist() == [0, 2]                     # 4 reference candidates, the 2 lowest indices
    assert all(1 not in p.tolist() for p in peers)
    assert peers[0].tolist() == [2, 3]


def test_peer_reference_separates_an_attacker_from_honest_clients_with_unusual_data():
    """Two honest data groups (8 + 4 clients, different classes, different update directions) and three
    attackers holding the majority group's classes but pushing a third direction. Against the global median the
    4 honest minority-group clients look as unusual as the attackers; against their peers they look normal, and
    the attackers stay at the bottom."""
    layout, model = _layout()
    dim = sum(v.numel() for v in model.state_dict().values())
    g = torch.Generator().manual_seed(0)
    a, b, evil = (torch.randn(dim, generator=g) for _ in range(3))
    ups = [a + 0.6 * torch.randn(dim, generator=g) for _ in range(8)]
    ups += [b + 0.6 * torch.randn(dim, generator=g) for _ in range(4)]
    ups += [0.5 * a + evil + 0.6 * torch.randn(dim, generator=g) for _ in range(3)]
    masks = torch.zeros(15, 20, dtype=torch.bool)
    masks[:8, :10] = True
    masks[8:12, 10:] = True
    masks[12:, :10] = True
    z_med = robust_z(screening_scores(ups, layout, masks, ScreenOptions("weights", "median")))
    z_peer = robust_z(screening_scores(ups, layout, masks, ScreenOptions("weights", "peer", peer_share=1 / 3)))
    assert (z_med[8:12] < -2.5).all()                       # honest minority blamed by the global median
    assert (z_peer[:12] > -2.5).all() and (z_peer[12:] < -2.5).all()
    # A wider peer set (10 of 14) holds only 3 of the minority's kind: it is judged by the majority again.
    z_wide = robust_z(screening_scores(ups, layout, masks, ScreenOptions("weights", "peer", peer_share=2 / 3)))
    assert (z_wide[8:12] < -2.5).all()


def test_colluders_reporting_one_mask_do_not_become_each_others_reference():
    """Three colluders (of 15) report identical masks, so each picks the other two first; with peer_share 1/3 the
    other two are 2 of 5 peers and the peer median stays honest. Five colluders take over (4 of 5 peers): the
    peer reference then hides them, a known limit (docstring of peer_sets)."""
    layout, model = _layout()
    dim = sum(v.numel() for v in model.state_dict().values())
    g = torch.Generator().manual_seed(1)
    honest, evil = torch.randn(dim, generator=g), torch.randn(dim, generator=g)
    masks = torch.rand(15, 20, generator=g) < 0.6
    for f, share, caught in ((3, 1 / 3, True), (5, 1 / 3, False)):
        ups = [honest + 0.8 * torch.randn(dim, generator=g) for _ in range(15 - f)]
        ups += [evil + 0.05 * torch.randn(dim, generator=g) for _ in range(f)]
        m = masks.clone()
        m[15 - f:] = m[15 - f]
        peers = peer_sets(m, share)
        assert set(peers[15 - f].tolist()) >= set(range(16 - f, 15))
        z = robust_z(screening_scores(ups, layout, m, ScreenOptions("weights", "peer", peer_share=share)))
        assert bool((z[15 - f:] < -2.5).all()) is caught


def test_peer_references_are_coordinate_medians_of_each_peer_set():
    vecs = torch.arange(12.0).reshape(4, 3)
    refs = peer_references(vecs, [torch.tensor([1, 2, 3]), torch.tensor([0]), torch.tensor([0, 1, 3]),
                                  torch.tensor([2])])
    assert torch.equal(refs[0], vecs[2]) and torch.equal(refs[1], vecs[0]) and torch.equal(refs[3], vecs[2])
    assert torch.equal(refs[2], vecs[1])


def test_critical_layers_with_the_peer_reference_match_a_manual_computation():
    stack, layout = _layered(n=9)
    masks = torch.rand(9, 6, generator=torch.Generator().manual_seed(2)) < 0.5
    opts = ScreenOptions("critical", "peer", critical_k=1, norm_bound=1e9)
    got = critical_layer_scores(list(stack), layout, opts, class_masks=masks)
    refs = peer_references(stack, peer_sets(masks, opts.peer_share))
    cos = torch.stack([torch.nn.functional.cosine_similarity(stack[:, s:e], refs[:, s:e], dim=1)
                       for _, s, e in layout.params], dim=1)
    top = cos.std(dim=0, correction=0).argmax()
    assert torch.allclose(got, cos[:, top], atol=1e-5)


def test_peer_reference_needs_class_masks_and_is_rejected_where_a_single_reference_is_used():
    stack, layout = _layered(n=6)
    with pytest.raises(ValueError):
        screening_scores(list(stack), layout, None, ScreenOptions("weights", "peer"))
    with pytest.raises(ValueError):
        ScreenOptions(shadow_reference="nearest")


def test_shadow_scores_are_logged_and_change_nothing(tmp_path):
    """A shadow reference is scored and logged for both signals of the union screen; the run itself (scores, z,
    decisions, accuracy) is identical to the run without it."""
    over = {"plus.reference": "median", "plus.signal": "weights_critical", "plus.two_sided": True,
            "plus.gap_gate": 8.0, "attack.kind": "backdoor"}
    plain = run_experiment(_cfg(tmp_path, name="plain", **over))
    shadow = run_experiment(_cfg(tmp_path, name="shadow", **over, **{"plus.shadow_reference": "peer"}))
    for d0, d1 in zip(plain["diagnostics"], shadow["diagnostics"], strict=True):
        assert d0["z"] == d1["z"] and d0["excluded"] == d1["excluded"]
        stats = d1["score_stats"]
        assert len(stats["shadow_scores"]) == len(d1["z"]) == len(stats["critical"]["shadow_scores"])
        assert "shadow_gap_ratio" in stats
    assert plain["final"] == shadow["final"]


def test_peer_reference_runs_end_to_end_and_leaves_out_sign_flippers(tmp_path):
    """9 clients, so the 2 attackers stay below half of a client's 3 peers (with 6 clients they would not)."""
    result = run_experiment(_cfg(tmp_path, name="peer", **{"data.num_clients": 9, "data.train_subset": 900,
                                                          "plus.reference": "peer", "plus.signal": "weights",
                                                          "plus.shadow_reference": "median"}))
    attackers = set(result["attackers"])
    assert all(attackers <= set(d["flagged"]) for d in result["detection"][1:])
