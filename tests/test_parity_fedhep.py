"""FedHEP parity with the authors' implementation. Each test runs their function and ours on the same input.
Skipped when the reference folder isn't present (set RFL_REFERENCE, default ../Topology-aware-FDL)."""
import importlib
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from rfl.aggregation.momentum import adaptive_momentum_beta
from rfl.client.fedhep import (
    FedHEPTrainer,
    HeadMemory,
    binomial_head_weights,
    confidence_head_weights,
    ensemble_logits,
    label_stats,
)
from rfl.client.sgd import SGDSettings
from rfl.core.state import flatten
from rfl.core.types import ClientData
from rfl.grouping.similarity import (
    count_misaligned,
    kmeans_on_updates,
    mean_member_similarity,
    random_assignment,
    random_project,
)
from rfl.models.resnet9 import MultiHeadResNet9

REF = Path(os.environ.get("RFL_REFERENCE", Path(__file__).resolve().parents[2] / "Topology-aware-FDL"))
pytestmark = pytest.mark.skipif(not (REF / "src" / "core" / "updater.py").exists(),
                                reason=f"reference code not found at {REF}")
C = 100


@pytest.fixture(scope="module")
def ref():
    sys.path.insert(0, str(REF))
    try:
        yield types.SimpleNamespace(
            model=importlib.import_module("src.core.model"),
            loss=importlib.import_module("src.core.loss"),
            updater=importlib.import_module("src.core.updater"),
            engine=importlib.import_module("src.core.hierarchical_ensemble_engine"),
            topo=importlib.import_module("src.topologies.hierarchical"),
            dataset=importlib.import_module("src.data.dataset"),
            config=importlib.import_module("src.config"),
        )
    finally:
        sys.path.remove(str(REF))


@pytest.fixture(scope="module")
def updater(ref):
    torch.manual_seed(0)
    return ref.updater.PyTorchLocalUpdater(device="cpu", in_channels=3, model_name="resnet9", num_classes=C)


class _Labelled:
    """What the authors' label statistics and data loader read from a client dataset."""

    def __init__(self, images, labels):
        self.images, self.labels = images, labels

    def __len__(self):
        return len(self.labels)


def _skewed_labels(seed, n=300):
    rng = np.random.RandomState(seed)
    classes = rng.choice(C, rng.randint(1, 30), replace=False)
    return torch.as_tensor(rng.choice(classes, n, p=rng.dirichlet(np.ones(len(classes)))))


def test_multihead_same_structure_and_same_init(ref):
    torch.manual_seed(42)
    theirs = ref.model.MultiHeadResNet9(in_channels=3, num_classes=C).state_dict()
    torch.manual_seed(42)
    ours = MultiHeadResNet9(3, C).state_dict()
    assert list(ours) == list(theirs)
    for k in ours:
        assert torch.equal(ours[k], theirs[k]), k


@pytest.mark.parametrize("seed", range(8))
def test_label_stats_identical(updater, seed):
    y = _skewed_labels(seed)
    r_t, mask_t, counts_t = updater._compute_label_stats(_Labelled(None, y), C)
    r_o, mask_o, counts_o = label_stats(y, C)
    assert r_o == r_t and torch.equal(mask_o, mask_t) and torch.equal(counts_o, counts_t)


@pytest.mark.parametrize("r", [0.0, 0.03, 0.2, 0.5, 0.85, 1.0])
@pytest.mark.parametrize("local_classes", [1, 7, 100])
def test_binomial_weights_identical(ref, r, local_classes):
    theirs = ref.loss.compute_dynamic_binomial_loss_weights(r, num_classes=C, local_classes=local_classes,
                                                            num_clusters=3)[3:]
    assert binomial_head_weights(r, C, local_classes, 3) == theirs


@pytest.mark.parametrize("seed", range(5))
def test_confidence_head_weights_identical(ref, seed):
    g = torch.Generator().manual_seed(seed)
    zr, zp, zl = (torch.randn(64, C, generator=g) * s for s in (1.0, 2.0, 3.0))
    mem = HeadMemory(ensemble_alpha=[0.2, 0.3, 0.5], head_losses={"root": 2.1, "parent": 1.3, "local": 0.4})
    theirs = ref.engine.compute_head_weights("dynamic_confidence", zr, zp, zl, mem.head_losses, mem.ensemble_alpha,
                                             1.0, 1.0, None)
    assert confidence_head_weights(zl, zp, zr, mem) == theirs


@pytest.mark.parametrize("seed", [42, 49, 7])
def test_random_assignment_identical(ref, seed):
    topo = ref.topo.HierarchicalTopology(num_clusters=3)
    topo.build(15, seed=seed)
    assert random_assignment(15, 3, seed) == {c: -2 - h for c, h in topo.client_to_head.items()}


def test_chunked_projection_identical(ref):
    v = np.random.RandomState(0).randn(15, 5000).astype(np.float32)
    np.testing.assert_allclose(random_project(v, 256, 11, chunk_rows=777), ref.topo._random_project(v, 256, 11),
                               rtol=1e-10, atol=1e-10)


def _deltas(seed, n=15, d=4000, noise=0.8):
    """Three directions plus noise. Low noise: clear groups. High noise: k-means needs all its iterations."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(3, d, generator=g)
    return {c: base[c % 3] + noise * torch.randn(d, generator=g) for c in range(n)}


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("noise", [0.8, 3.0, 10.0])
def test_kmeans_grouping_identical(ref, seed, noise):
    deltas = _deltas(seed, noise=noise)
    topo = ref.topo.HierarchicalTopology(num_clusters=3)
    topo.build_update_similarity(15, deltas, seed=seed + 100)
    assert kmeans_on_updates(deltas, 3, seed + 100) == {c: -2 - h for c, h in topo.client_to_head.items()}


def _fake_engine(ref, assignment, num_groups=3):
    topo = types.SimpleNamespace(get_neighbors=lambda c: [-2 - assignment[c]])
    fake = types.SimpleNamespace(topology=topo, misalignment_threshold=0.15,
                                 cluster_heads_state={-2 - g: None for g in range(num_groups)})
    E = ref.engine.HierarchicalEnsembleEngine
    fake._client_cluster_affinities = lambda cl, d: E._client_cluster_affinities(fake, cl, d)
    return fake


@pytest.mark.parametrize("seed", range(6))
def test_group_similarity_measures_identical(ref, seed):
    deltas = _deltas(seed)
    assignment = random_assignment(15, 3, seed)
    fake = _fake_engine(ref, assignment)
    E = ref.engine.HierarchicalEnsembleEngine
    ids = list(range(15))
    assert mean_member_similarity(assignment, deltas, 3) == pytest.approx(
        E._avg_intra_cluster_similarity(fake, ids, deltas), abs=1e-12)
    assert count_misaligned(assignment, deltas, 3, 0.15) == E._count_misaligned_clients(fake, ids, deltas)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("members", [1, 2, 5])
@pytest.mark.parametrize("offset", [0.0, 0.5, 5.0])  # beta near the 0.9 clip, in between, near the 0.1 clip
def test_adaptive_momentum_identical(ref, seed, members, offset):
    g = torch.Generator().manual_seed(seed)
    old, new = torch.randn(500, generator=g), torch.randn(500, generator=g)
    member_deltas = [torch.randn(500, generator=g) * (1 + i) + offset for i in range(members)]
    head = types.SimpleNamespace(weights=old.clone())
    fake = types.SimpleNamespace(cluster_heads_state={0: head},
                                 config=types.SimpleNamespace(clients=types.SimpleNamespace(cluster_momentum_beta=0.0)))
    ref.engine.HierarchicalEnsembleEngine._apply_cluster_momentum(fake, 0, new, intra_cluster_deltas=member_deltas)
    beta = adaptive_momentum_beta(member_deltas)
    ours = beta * old + (1 - beta) * new if beta > 0 else new
    assert torch.allclose(ours, head.weights, atol=1e-6)


def _client(seed, n=70):
    g = torch.Generator().manual_seed(seed)
    y = _skewed_labels(seed, n)
    return torch.randn(n, 3, 32, 32, generator=g), y


@pytest.mark.parametrize("seed", [0, 1])
def test_local_training_round_identical(ref, updater, seed):
    """One full local round (5 epochs, all three heads, alpha updates) from identical weights and the same torch
    seed: their `_update_hep` and our trainer must end at the same weights, losses and ensemble alpha."""
    x, y = _client(seed)
    torch.manual_seed(123)
    start = MultiHeadResNet9(3, C)
    cfg = ref.config.ClientConfig(num_clients=15, model_name="resnet9")
    state = ref.updater.ClientState(0, ref.model.model_to_vector(start))
    torch.manual_seed(seed)
    theirs = updater._update_hep(state, _Labelled(x, y), cfg, ref.dataset.FastTensorDataLoader(x, y, 32, True),
                                 0.05, None)
    model, mem = MultiHeadResNet9(3, C), HeadMemory()
    model.load_state_dict(start.state_dict())
    torch.manual_seed(seed)
    FedHEPTrainer(SGDSettings(local_epochs=5), total_rounds=1, num_classes=C, num_groups=3).train(
        model, ClientData(x, y, x[:1], y[:1]), 1, mem)
    assert torch.allclose(flatten(model.state_dict()), theirs.weights, atol=1e-5)
    assert mem.ensemble_alpha == pytest.approx(theirs.ensemble_alpha, abs=1e-6)
    assert mem.head_losses == pytest.approx(theirs.head_losses, abs=1e-5)
    assert mem.r_skew == theirs.r_skew


@pytest.mark.parametrize("local_loss", [0.5, -6.0])  # -6 (not a real loss) gives the down-scaled local head weight
def test_ensemble_prediction_identical(ref, updater, local_loss):
    """Their `evaluate_ensemble` (shared-backbone branch) and our `ensemble_logits` agree, for a skewed client
    (blended heads) and a near-uniform one (root head only). Compared on per-client accuracy (masked prediction)
    and on the test loss, which they compute on the unmasked blend and which moves with every blend weight and
    temperature."""
    torch.manual_seed(5)
    model = MultiHeadResNet9(3, C).eval()
    with torch.no_grad():  # heads at different scales, so both temperature clamps (0.2 and 2.0) are active
        for p, scale in ((model.fc2_local, 0.01), (model.fc2_parent, 5.0)):
            p.weight.mul_(scale)
            p.bias.mul_(scale)
    g = torch.Generator().manual_seed(9)
    # client 1: 90 classes near-uniform, r_skew ~0.89, just above the 0.85 root-only threshold
    clients = {0: _skewed_labels(3, 300), 1: torch.randint(0, 90, (2000,), generator=g)}
    images = {c: torch.randn(len(y), 3, 32, 32, generator=g) for c, y in clients.items()}
    states, ours = {}, {}
    loss_sum = 0.0
    for c, y in clients.items():
        r, mask, _ = label_stats(y, C)
        losses = {"root": 3.0, "parent": 1.0, "local": local_loss}
        mem = HeadMemory(ensemble_alpha=[0.5, 0.3, 0.2], head_losses=losses, r_skew=r, active_mask=mask)
        s = ref.updater.ClientState(c, ref.model.model_to_vector(model))
        s.ensemble_alpha, s.head_losses, s.r_skew, s.active_mask = mem.ensemble_alpha, mem.head_losses, r, mask
        states[c] = s
        unmasked = HeadMemory(mem.ensemble_alpha, None, mem.head_losses, r, None)
        with torch.no_grad():
            pred = ensemble_logits(model, images[c], mem).argmax(1)
            loss_sum += torch.nn.functional.cross_entropy(ensemble_logits(model, images[c], unmasked), y,
                                                          reduction="sum").item()
        ours[c] = 100.0 * int((pred == y).sum()) / len(y)
    clients_cfg = ref.config.ClientConfig(num_clients=2, model_name="resnet9",
                                          compute_optimization_mode="shared_backbone")
    E = ref.engine.HierarchicalEnsembleEngine
    fake = types.SimpleNamespace(config=types.SimpleNamespace(clients=clients_cfg), updater=updater, device="cpu",
                                 clients_state=states,
                                 client_test_datasets={c: _Labelled(images[c], y) for c, y in clients.items()})
    fake._safr_blend_weights = lambda *a: E._safr_blend_weights(fake, *a)
    fake._apply_top2_routing = lambda *a: E._apply_top2_routing(fake, *a)
    _, their_loss = E.evaluate_ensemble(fake, 5)
    assert 0.95 > states[1].r_skew >= 0.85 > states[0].r_skew  # both branches exercised
    for c in clients:
        assert ours[c] == pytest.approx(fake._last_per_client_accuracy[c], abs=1e-9)
    assert loss_sum / sum(len(y) for y in clients.values()) == pytest.approx(their_loss, rel=1e-5)
