"""Baseline parity with the authors' implementation: two federated rounds of one client with each local rule,
compared against their updaters (same start, same torch seed), and Multi-Krum's selection.
Skipped when the reference folder isn't present (set RFL_REFERENCE, default ../Topology-aware-FDL)."""
import importlib
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from rfl.aggregation.multikrum import MultiKrum, krum_scores
from rfl.client.ditto import DittoTrainer
from rfl.client.fedprox import FedProxTrainer
from rfl.client.fedrep import FedRepTrainer
from rfl.client.scaffold import ScaffoldTrainer
from rfl.client.sgd import SGDSettings, SGDTrainer, cosine_lr
from rfl.core.state import flatten, unflatten
from rfl.core.types import ClientData, ClientUpdate
from rfl.models.resnet9 import ResNet9

REF = Path(os.environ.get("RFL_REFERENCE", Path(__file__).resolve().parents[2] / "Topology-aware-FDL"))
pytestmark = pytest.mark.skipif(not (REF / "src" / "baselines" / "scaffold_updater.py").exists(),
                                reason=f"reference code not found at {REF}")
C, ROUNDS = 100, 2


@pytest.fixture(scope="module")
def ref():
    sys.path.insert(0, str(REF))
    try:
        yield types.SimpleNamespace(
            model=importlib.import_module("src.core.model"),
            updater=importlib.import_module("src.core.updater"),
            fedprox=importlib.import_module("src.baselines.fedprox_updater"),
            scaffold=importlib.import_module("src.baselines.scaffold_updater"),
            krum=importlib.import_module("src.baselines.multi_krum_aggregator"),
            bconfig=importlib.import_module("src.baselines.config"),
        )
    finally:
        sys.path.remove(str(REF))


class _Labelled:
    def __init__(self, images, labels):
        self.images, self.labels = images, labels

    def __len__(self):
        return len(self.labels)


def _client(seed=0, n=70):
    rng = np.random.RandomState(seed)
    y = torch.as_tensor(rng.choice(rng.choice(C, 25, replace=False), n))
    return torch.randn(n, 3, 32, 32, generator=torch.Generator().manual_seed(seed)), y


METHODS = {  # name -> (authors' personalization_method, updater class name, our rule factory)
    "fedavg": ("none", "updater", lambda s: SGDTrainer(s, ROUNDS)),
    "fedprox": ("fedprox", "fedprox", lambda s: FedProxTrainer(s, ROUNDS, 0.01)),
    "ditto": ("ditto", "updater", lambda s: DittoTrainer(s, ROUNDS, 0.05)),
    "fedrep": ("fedrep", "updater", lambda s: FedRepTrainer(s, ROUNDS, 1)),
    "scaffold": ("scaffold", "scaffold", lambda s: ScaffoldTrainer(s, ROUNDS)),
}


def _their_updater(ref, kind):
    torch.manual_seed(0)
    cls = {"updater": ref.updater.PyTorchLocalUpdater, "fedprox": ref.fedprox.FedProxUpdater,
           "scaffold": ref.scaffold.ScaffoldUpdater}[kind]
    return cls(device="cpu", in_channels=3, model_name="resnet9", num_classes=C)


@pytest.mark.parametrize("method", list(METHODS))
@pytest.mark.parametrize("flip", [False, True])
def test_two_rounds_identical(ref, method, flip):
    """Round 1 from a fresh model, round 2 from the round-1 upload (a one-client federation), so persistent
    client state (Ditto's personal model, FedRep's head, SCAFFOLD's control variates) is exercised."""
    pers, kind, make_rule = METHODS[method]
    x, y = _client()
    torch.manual_seed(123)
    start = ResNet9(3, C)
    template = start.state_dict()

    upd = _their_updater(ref, kind)
    cfg = ref.bconfig.BaselineClientConfig(num_clients=1, model_name="resnet9", local_steps=3,
                                           personalization_method=pers, fedprox_mu=0.01, ditto_lambda=0.05,
                                           fedrep_head_epochs=1)
    state = ref.updater.ClientState(0, ref.model.model_to_vector(start))
    state.is_byzantine, state.byzantine_type = flip, "label_flip"
    server_c = torch.zeros(sum(p.numel() for p in start.parameters()))
    theirs = []
    for r in range(1, ROUNDS + 1):
        torch.manual_seed(r)
        lr = cosine_lr(r, ROUNDS, 0.05, 0.001)
        if method == "scaffold":
            state = upd.update_scaffold(state.copy(), _Labelled(x, y), cfg, None, lr, server_c)
            server_c = server_c + state._scaffold_delta_c
        else:
            state = upd.update(state.copy(), _Labelled(x, y), cfg, None, lr)
        personal = None if state.local_weights is None else state.local_weights.clone()
        theirs.append((state.weights.clone(), personal))

    rule = make_rule(SGDSettings(local_epochs=3))
    model = ResNet9(3, C)
    vec = flatten(template)
    targets = (C - 1) - y if flip else None
    for r in range(1, ROUNDS + 1):
        model.load_state_dict(unflatten(vec, template))
        torch.manual_seed(r)
        rule.train(model, ClientData(x, y, x[:1], y[:1]), r, 0, targets)
        vec = rule.upload(model, 0)
        rule.end_round()
        their_upload, their_personal = theirs[r - 1]
        assert torch.allclose(vec, their_upload, atol=1e-5), f"round {r} upload"
        if method in ("ditto", "fedrep"):
            assert torch.allclose(rule.personal_model(0), their_personal, atol=1e-5), f"round {r} personal"


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("f", [0, 2])
def test_multikrum_selection_identical(ref, seed, f):
    """Their aggregator works on whole model vectors (start + update); ours on updates."""
    g = torch.Generator().manual_seed(seed)
    start = torch.randn(3000, generator=g) * 3
    deltas = [torch.randn(3000, generator=g) * (1 + 0.2 * i) for i in range(15)]
    deltas[3] = -4 * deltas[3]
    deltas[9] = deltas[9] + 5
    states = [types.SimpleNamespace(weights=start + d, is_byzantine=i in (3, 9)) for i, d in enumerate(deltas)]
    if f == 0:  # no attacker flags: their f falls back to (n - 3) // 2
        for s in states:
            s.is_byzantine = False
    theirs = ref.krum.MultiKrumAggregator(f=0, m=3).aggregate(states)
    ours = MultiKrum(3, 2 if f else None)
    step = ours.aggregate([ClientUpdate(i, 1, {"state": d}) for i, d in enumerate(deltas)])["state"]
    assert torch.allclose(start + step, theirs, atol=1e-4)


def test_scaffold_two_clients_identical(ref):
    """Two clients over three rounds, so control variates actually correct the gradients (with one client the
    correction c - c_i is always zero). Server: c += mean(delta_c_i); next start = plain mean of uploads.

    The control variate divides a weight difference by (steps x lr), and SCAFFOLD's training amplifies tiny
    differences: computing x / s instead of the authors' (1 / s) * x (one rounding step apart) put uploads 3.6e-2
    apart by round 2. A 10-round learning-rate schedule keeps lr high so that sensitivity is exercised."""
    data = [_client(0), _client(1, n=50)]
    torch.manual_seed(123)
    start = ResNet9(3, C)
    template = start.state_dict()
    upd = _their_updater(ref, "scaffold")
    cfg = ref.bconfig.BaselineClientConfig(num_clients=2, model_name="resnet9", local_steps=3,
                                           personalization_method="scaffold")
    states = [ref.updater.ClientState(i, ref.model.model_to_vector(start)) for i in range(2)]
    server_c = torch.zeros(sum(p.numel() for p in start.parameters()))
    rule, model = ScaffoldTrainer(SGDSettings(local_epochs=3), 10), ResNet9(3, C)
    theirs_vec = ours_vec = flatten(template)
    for r in range(1, 4):
        lr = cosine_lr(r, 10, 0.05, 0.001)
        their_up, our_up, dcs = [], [], []
        for i, (x, y) in enumerate(data):
            torch.manual_seed(10 * r + i)
            states[i].weights = theirs_vec.clone()
            states[i] = upd.update_scaffold(states[i].copy(), _Labelled(x, y), cfg, None, lr, server_c)
            dcs.append(states[i]._scaffold_delta_c)
            their_up.append(states[i].weights.clone())
            model.load_state_dict(unflatten(ours_vec, template))
            torch.manual_seed(10 * r + i)
            rule.train(model, ClientData(x, y, x[:1], y[:1]), r, i)
            our_up.append(rule.upload(model, i))
        server_c = server_c + torch.stack(dcs).mean(dim=0)
        rule.end_round()
        for i in range(2):
            gap = (our_up[i] - their_up[i]).abs().max().item()
            assert gap < 1e-4, f"round {r} client {i}: max difference {gap:.2e}"
        theirs_vec, ours_vec = torch.stack(their_up).mean(dim=0), torch.stack(our_up).mean(dim=0)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("f", [1, 3, 6, 9])
def test_krum_scores_identical(ref, seed, f):
    g = torch.Generator().manual_seed(seed)
    stacked = torch.randn(15, 500, generator=g) * torch.linspace(0.5, 3, 15).unsqueeze(1)
    assert torch.allclose(krum_scores(stacked, f), ref.krum.multi_krum_scores(stacked, f), rtol=1e-5)
