"""Defended FedHEP parity with the authors' implementation: attacker choice, the four attacks, SCCF aggregation.
Skipped when the reference folder isn't present (set RFL_REFERENCE, default ../Topology-aware-FDL)."""
import importlib
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from rfl.attacks.byzantine import ByzantineAttack, select_attackers
from rfl.client.fedhep import FedHEPTrainer, HeadMemory
from rfl.client.sgd import SGDSettings
from rfl.core.state import flatten
from rfl.core.types import ClientData
from rfl.defense.soft_cosine import SoftCosineAggregator, flat_layout, soft_cosine_trust
from rfl.models.resnet9 import MultiHeadResNet9

REF = Path(os.environ.get("RFL_REFERENCE", Path(__file__).resolve().parents[2] / "Topology-aware-FDL"))
pytestmark = pytest.mark.skipif(not (REF / "src" / "core" / "aggregator.py").exists(),
                                reason=f"reference code not found at {REF}")
C = 100


@pytest.fixture(scope="module")
def ref():
    sys.path.insert(0, str(REF))
    try:
        yield types.SimpleNamespace(
            model=importlib.import_module("src.core.model"),
            agg=importlib.import_module("src.core.aggregator"),
            updater=importlib.import_module("src.core.updater"),
            rnd=importlib.import_module("src.utils.random"),
            dataset=importlib.import_module("src.data.dataset"),
            config=importlib.import_module("src.config"),
        )
    finally:
        sys.path.remove(str(REF))


@pytest.mark.parametrize("seed", [42, 123, 7, 0])
@pytest.mark.parametrize("rate", [0.1, 0.2, 0.3, 0.4])
def test_attacker_choice_identical(ref, seed, rate):
    """Their BaseEngine: set_seed, then get_random_state() (first call), then rand() < rate per client."""
    ref.rnd.set_seed(seed)
    rng = ref.rnd.get_random_state()
    theirs = frozenset(c for c in range(15) if rng.rand() < rate)
    assert select_attackers(15, rate, seed) == theirs


@pytest.mark.parametrize("kind", ["sign_flip", "gradient_ascent", "random_noise", "label_flip"])
def test_upload_attacks_identical(ref, kind):
    g = torch.Generator().manual_seed(1)
    start, trained = torch.randn(1000, generator=g), torch.randn(1000, generator=g)
    state = types.SimpleNamespace(is_byzantine=True, byzantine_type=kind, weights=trained.clone(),
                                  parent_weights=None, local_weights=None)
    torch.manual_seed(3)
    theirs = ref.updater.PyTorchLocalUpdater._apply_byzantine_attack(None, state, start).weights
    torch.manual_seed(3)
    assert torch.equal(ByzantineAttack(kind).poison_upload(trained, start), theirs)


def test_label_flip_local_round_identical(ref):
    """A label-flipping attacker's full local round, as in their `_update_hep` with is_byzantine set."""
    torch.manual_seed(0)
    updater = ref.updater.PyTorchLocalUpdater(device="cpu", in_channels=3, model_name="resnet9", num_classes=C)
    rng = np.random.RandomState(4)
    y = torch.as_tensor(rng.choice(rng.choice(C, 40, replace=False), 70))
    x = torch.randn(70, 3, 32, 32, generator=torch.Generator().manual_seed(4))
    torch.manual_seed(123)
    start = MultiHeadResNet9(3, C)
    state = ref.updater.ClientState(0, ref.model.model_to_vector(start))
    state.is_byzantine, state.byzantine_type = True, "label_flip"
    labelled = type("L", (), {"images": x, "labels": y, "__len__": lambda self: 70})()
    torch.manual_seed(1)
    theirs = updater._update_hep(state, labelled, ref.config.ClientConfig(num_clients=15, model_name="resnet9"),
                                 ref.dataset.FastTensorDataLoader(x, y, 32, True), 0.05, None)
    model, mem = MultiHeadResNet9(3, C), HeadMemory()
    model.load_state_dict(start.state_dict())
    torch.manual_seed(1)
    flipped = ByzantineAttack("label_flip").poison_labels(y, C)
    FedHEPTrainer(SGDSettings(local_epochs=5), 1, C, 3).train(model, ClientData(x, y, x[:1], y[:1]), 1, mem,
                                                              targets=flipped)
    assert torch.allclose(flatten(model.state_dict()), theirs.weights, atol=1e-5, equal_nan=True)
    assert mem.head_losses == pytest.approx(theirs.head_losses, rel=1e-5)


def _updates(case, n, dim, g):
    """Honest updates around a shared direction, plus the case's attackers in the last rows."""
    base = torch.randn(dim, generator=g)
    ups = [base + 0.7 * torch.randn(dim, generator=g) for _ in range(n)]
    if case == "sign_flip":
        ups[-2:] = [-1.5 * u for u in ups[-2:]]
    elif case == "scaled":
        ups[-1] = 50.0 * ups[-1]
    elif case == "no_consensus":
        ups = [torch.randn(dim, generator=g) for _ in range(n)]
    return ups


@pytest.mark.parametrize("tier", ["parent", "global"])
@pytest.mark.parametrize("case", ["honest", "sign_flip", "scaled", "no_consensus"])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("n", [1, 4, 7])
def test_soft_cosine_aggregation_identical(ref, tier, case, masked, n):
    """Their DeltaSpaceRobustAggregator with the settings UnifiedDefendedEngine gives each tier, on a small
    multi-head model's flat layout (buffers -> median, weights -> bounded trust-weighted sum)."""
    classes = 20
    model = MultiHeadResNet9(3, classes, base_channels=4)
    state = model.state_dict()
    dim = sum(v.numel() for v in state.values())
    g = torch.Generator().manual_seed(n * 10 + len(case))
    ups = _updates(case, n, dim, g)
    masks = torch.rand(n, classes, generator=g) < 0.4 if masked else None
    k, hard = (2.0, False) if tier == "parent" else (3.0, True)
    theirs_agg = ref.agg.DeltaSpaceRobustAggregator(
        mode="soft_cosine", temperature=0.5, norm_bound_k=k, adaptive_temperature=True, hard_rejection=hard,
        hard_threshold=-0.1 if tier == "global" else 0.0, buffer_slices=ref.agg.compute_buffer_slices(model),
        classifier_slice=ref.agg.compute_classifier_slices(model))
    theirs = theirs_agg.aggregate_deltas(ups, reference=None, active_masks=masks)
    layout = flat_layout(state, {name for name, _ in model.named_parameters()})
    ours, trust = SoftCosineAggregator(layout, k, hard, -0.1 if tier == "global" else 0.0).aggregate(ups, masks)
    assert torch.allclose(trust, theirs_agg.get_last_trust_scores(), atol=1e-6)
    assert torch.allclose(ours, theirs, atol=1e-5)


@pytest.mark.parametrize("seed", range(80))
def test_trust_weights_identical_across_regimes(ref, seed):
    """Their `soft_cosine_trust` on random sender mixes (alignment spread from opposing to agreeing, noise over
    1.5 decades). Across the seeds this crosses the no-consensus cut (0.2), the temperature switch (cosine
    variance 0.05) and the hard-rejection threshold (-0.1) from both sides."""
    g = torch.Generator().manual_seed(seed)
    n = int(torch.randint(2, 40, (1,), generator=g))
    mu, spread = float(torch.rand(1, generator=g)) * 1.5 - 0.3, float(torch.rand(1, generator=g)) * 1.5
    noise = 10 ** (float(torch.rand(1, generator=g)) * 2 - 1.5)
    base = torch.randn(300, generator=g)
    align = mu + spread * (torch.rand(n, generator=g) - 0.5)
    ups = torch.stack([base * a + noise * torch.randn(300, generator=g) for a in align])
    hard = seed % 2 == 0
    theirs = ref.agg.soft_cosine_trust(ups, temperature=0.5, hard_rejection=hard, hard_threshold=-0.1,
                                       adaptive_temperature=True)
    assert torch.allclose(soft_cosine_trust(ups, 0.5, hard, -0.1, None, None), theirs, atol=1e-7)
