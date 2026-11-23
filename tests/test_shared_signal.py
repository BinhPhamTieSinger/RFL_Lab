"""Phase 8: the "shared" screening signal, the whole update without the personal heads (fc2_parent, fc2_local).
Members of one parent share its parent head, so with the personal heads in, honest clients split by group."""
import torch

from rfl.defense.screening import PERSONAL_HEADS, ScreenOptions, robust_z, screening_scores
from rfl.defense.soft_cosine import FlatLayout
from rfl.experiment.runner import run_experiment
from tests.test_fedhep_plus import _cfg


def _toy(seed=0):
    """15 clients in 3 parent groups of 4 honest members each plus 3 attackers in group 0. Honest updates share the
    trunk and root-head direction; each group adds its own parent-head direction, small for groups 0 and 1 and
    large for group 2 (one group's parent head drifting away, as in the logged runs); attackers add a trunk
    direction."""
    g = torch.Generator().manual_seed(seed)
    sizes = {"trunk.conv": 3000, "fc2_root.weight": 400, "fc2_parent.weight": 1500, "fc2_local.weight": 400}
    params, s = [], 0
    for name, k in sizes.items():
        params.append((name, s, s + k))
        s += k
    layout = FlatLayout(torch.zeros(s, dtype=torch.bool), (3000, 3400), tuple(params))
    common = torch.randn(s, generator=g)
    group_dirs = [torch.zeros(s) for _ in range(3)]
    for d, size in zip(group_dirs, (0.3, 0.3, 3.0), strict=True):
        d[3400:4900] = size * torch.randn(1500, generator=g)
    evil = torch.zeros(s)
    evil[:3000] = 1.5 * torch.randn(3000, generator=g)
    ups, groups = [], []
    for k in range(12):
        ups.append(common + group_dirs[k % 3] + 0.4 * torch.randn(s, generator=g))
        groups.append(k % 3)
    for _ in range(3):
        ups.append(common + group_dirs[0] + evil + 0.4 * torch.randn(s, generator=g))
        groups.append(0)
    return ups, layout, groups


def test_shared_signal_leaves_out_exactly_the_personal_heads():
    ups, layout, _ = _toy()
    a = torch.zeros(len(ups[0]))
    b = a.clone()
    for name, s, e in layout.params:
        if name.startswith(PERSONAL_HEADS):
            b[s:e] = torch.randn(e - s)      # differ only in the personal heads
    base = torch.randn(len(a))
    stack = [base + a, base + b, base + a, base + b, base + a]
    shared = screening_scores(stack, layout, None, ScreenOptions("shared", "median", norm_bound=0))
    weights = screening_scores(stack, layout, None, ScreenOptions("weights", "median", norm_bound=0))
    assert torch.allclose(shared, torch.ones(5), atol=1e-5)
    assert weights.min() < 0.99


def test_whole_update_splits_honest_groups_and_the_shared_signal_does_not():
    ups, layout, groups = _toy()
    z_w = robust_z(screening_scores(ups, layout, None, ScreenOptions("weights", "median")))
    z_s = robust_z(screening_scores(ups, layout, None, ScreenOptions("shared", "median")))
    group2 = [k for k in range(12) if groups[k] == 2]
    # with the parent heads in, the whole drifting group sits far below the median
    assert (z_w[group2] < -2.5).all()
    # without them, honest clients are typical and the attackers stand out
    assert (z_s[:12].abs() < 2.5).all() and (z_s[12:] < -2.5).all()


def test_shared_critical_union_runs_end_to_end_and_leaves_out_sign_flippers(tmp_path):
    result = run_experiment(_cfg(tmp_path, name="shared", **{
        "plus.reference": "median", "plus.signal": "shared_critical", "plus.two_sided": True,
        "plus.global_rule": "arc_nnm_tm", "plus.global_f": 2}))
    attackers = set(result["attackers"])
    assert all(attackers <= set(d["flagged"]) for d in result["detection"][1:])
    stats = result["diagnostics"][1]["score_stats"]
    assert "critical" in stats and len(stats["critical_scores"]) == len(result["diagnostics"][1]["z"])
