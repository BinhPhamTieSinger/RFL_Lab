"""Phase 5: established robust rules (rfl.defense.classic) and FedHEP with each one in its Global step."""
import pytest
import torch

from rfl.defense.classic import bulyan_rule, flame_rule, fltrust_rule, median_rule, trimmed_mean_rule
from rfl.experiment.config import load_config
from rfl.experiment.runner import run_experiment


def _honest_and_attackers(seed: int = 0, n_honest: int = 12, n_bad: int = 3, dim: int = 200, scale: float = -5.0):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(dim, generator=g)
    honest = base + 0.3 * torch.randn(n_honest, dim, generator=g)
    bad = scale * (base + 0.3 * torch.randn(n_bad, dim, generator=g))
    return torch.cat([honest, bad]), base


def test_median_and_trimmed_mean_on_identical_updates_return_that_update():
    stack = torch.arange(6.0).repeat(7, 1)
    assert torch.equal(median_rule(stack).update, stack[0])
    assert torch.allclose(trimmed_mean_rule(stack, 3).update, stack[0])


def test_trimmed_mean_drops_k_extremes_per_coordinate():
    stack = torch.tensor([[0.0], [1.0], [2.0], [3.0], [100.0], [-100.0], [4.0]])
    assert trimmed_mean_rule(stack, 1).update.item() == pytest.approx(2.0)  # mean of 0..4
    with pytest.raises(ValueError):
        trimmed_mean_rule(stack, 4)


def test_bulyan_selects_honest_updates_and_needs_4f_plus_3():
    stack, base = _honest_and_attackers()
    r = bulyan_rule(stack, 3)
    assert set(r.accepted) <= set(range(12)) and len(r.accepted) == 15 - 6
    assert torch.nn.functional.cosine_similarity(r.update, base, dim=0) > 0.95
    with pytest.raises(ValueError):
        bulyan_rule(stack[:14], 3)


def test_bulyan_coordinate_step_averages_values_closest_to_the_median():
    # 7 updates, f = 1: Krum keeps theta = 5, then beta = 3 values per coordinate closest to their median.
    stack = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [50.0], [-50.0]])
    r = bulyan_rule(stack, 1)
    assert set(r.accepted) == {0, 1, 2, 3, 4}
    assert r.update.item() == pytest.approx(3.0)  # 2, 3, 4


@pytest.mark.parametrize("on", ["models", "updates"])
def test_flame_rejects_sign_flipped_updates_clips_to_the_median_norm_and_adds_noise(on):
    stack, base = _honest_and_attackers()
    start = torch.randn(200, generator=torch.Generator().manual_seed(9))
    r = flame_rule(stack, start, 0.0, torch.Generator().manual_seed(0), on=on)
    assert set(r.accepted) <= set(range(12)) and len(r.accepted) >= 15 // 2 + 1  # attackers out, a majority in
    bound = stack.norm(dim=1).median()
    assert r.update.norm() <= bound + 1e-5          # mean of updates clipped to S has norm <= S
    noisy = flame_rule(stack, start, 0.01, torch.Generator().manual_seed(0), on=on)
    assert (noisy.update - r.update).std().item() == pytest.approx(0.01 * bound.item(), rel=0.3)


def test_fltrust_zero_trust_for_opposed_updates_and_server_norm():
    stack, base = _honest_and_attackers()
    server = 0.5 * base
    r = fltrust_rule(stack, server)
    assert (r.scores[12:] == 0).all() and (r.scores[:12] > 0).all()
    assert set(r.accepted) == set(range(12))
    assert r.update.norm() <= server.norm() + 1e-5   # weighted mean of vectors of norm |g0|
    assert fltrust_rule(-stack[:12], server).update.abs().sum() == 0  # nobody trusted: no step


def _cfg(tmp_path, rule, **over):
    base = {"name": f"r_{rule}", "method": "fedhep_robust", "model": "resnet9_multihead",
            "data.dataset": "synthetic", "data.num_clients": 15, "data.train_subset": 1500,
            "data.test_subset": 300,
            "data.alpha": 0.5, "train.local_epochs": 1, "run.rounds": 2, "run.eval_every": 2, "run.device": "cpu",
            "run.seed": 3, "run.output_dir": str(tmp_path), "robust.rule": rule, "robust.root_size": 30,
            "attack.kind": "sign_flip", "attack.count": 3}
    base.update(over)
    return load_config(overrides=base)


@pytest.mark.parametrize("rule", ["median", "trimmed_mean", "bulyan", "flame", "fltrust"])
def test_each_rule_runs_inside_fedhep_and_reports_detection(tmp_path, rule):
    result = run_experiment(_cfg(tmp_path, rule))
    attackers = set(result["attackers"])
    first = result["detection"][0]  # the toy task is nearly learned by round 2, after which updates are noise
    assert len(attackers) == 3 and first["auroc"] is not None
    if rule in ("bulyan", "flame", "fltrust"):  # rules that pick clients leave the sign flippers out
        assert attackers <= set(first["flagged"])
    assert result["history"][-1]["pooled_acc"] > 0


def test_fltrust_server_training_leaves_client_randomness_untouched(tmp_path):
    """Same seed, same clients: the clients' first-round training must not depend on the server's own training,
    so the round-1 uploads under FLTrust and under the median rule are identical."""
    from rfl.data.clients import build_clients
    from rfl.data.partition import make_partitioner
    from rfl.experiment.registry import METHODS
    from rfl.experiment.runner import set_seed, synthetic_data
    from rfl.models.resnet9 import MultiHeadResNet9

    uploads = {}
    for rule in ("median", "fltrust"):
        cfg = _cfg(tmp_path, rule, **{"attack.kind": None})
        set_seed(cfg.run.seed)
        xtr, ytr, xte, yte = synthetic_data(1500, 300, 10, cfg.run.seed, torch.device("cpu"))
        clients = build_clients(xtr, ytr, xte, yte, make_partitioner(0.5), 15, cfg.run.seed)
        server = METHODS["fedhep_robust"](cfg, MultiHeadResNet9(3, 10), clients)
        server.run_round(1)
        uploads[rule] = torch.stack(server.trained)
    assert torch.equal(uploads["median"], uploads["fltrust"])
