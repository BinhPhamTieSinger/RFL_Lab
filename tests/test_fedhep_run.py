"""FedHEP server rules and an end-to-end run on synthetic data."""
import torch

from rfl.client.fedhep import HeadMemory
from rfl.core.state import flatten
from rfl.core.types import ClientData
from rfl.experiment.config import load_config
from rfl.experiment.runner import run_experiment
from rfl.grouping.similarity import Regroup, SimilarityGrouping
from rfl.models.resnet9 import MultiHeadResNet9
from rfl.server.fedhep_server import FedHEPServer


class ShiftTrainer:
    """Adds 1 to every float parameter (so all clients' updates point the same way) and keeps a local head."""

    def train(self, model, data, round_idx, memory: HeadMemory, targets=None):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        memory.local_head = {k: v.clone() for k, v in model.fc2_local.state_dict().items()}


def _clients(n):
    return [ClientData(torch.zeros(4, 3, 32, 32), torch.zeros(4, dtype=torch.long), torch.zeros(1, 3, 32, 32),
                       torch.zeros(1, dtype=torch.long)) for _ in range(n)]


def test_agreeing_group_copies_round_start_global_and_global_moves():
    model = MultiHeadResNet9(3, 10)
    start = flatten(model.state_dict())
    server = FedHEPServer(model, _clients(4), ShiftTrainer(), SimilarityGrouping(4, 2, seed=0))
    server.run_round(1)
    shift = server.trained[0] - start
    assert torch.allclose(server.global_vec, start + shift)  # every client moved the same way
    for g, parent in server.parents.items():  # identical updates: cosine 1 > 0.5, so parents copy the start model
        if server.grouping.members(g, range(4)):
            assert torch.allclose(parent, start)


def test_first_regroup_waits_for_stable_updates_or_round_ten():
    g = torch.Generator().manual_seed(0)
    grouping = SimilarityGrouping(6, 3, seed=1)
    events = [grouping.step({c: torch.randn(500, generator=g) for c in range(6)}, r) for r in range(1, 11)]
    assert events[:9] == [Regroup.NONE] * 9  # fresh random updates each round: never stable
    assert events[9] is Regroup.INITIAL and grouping.has_grouped


def test_stable_updates_trigger_regroup_at_round_two():
    g = torch.Generator().manual_seed(0)
    fixed = {c: torch.randn(500, generator=g) for c in range(6)}
    grouping = SimilarityGrouping(6, 3, seed=1)
    assert grouping.step(fixed, 1) is Regroup.NONE
    assert grouping.step(fixed, 2) is Regroup.INITIAL


def test_initial_regroup_drops_local_heads():
    server = FedHEPServer(MultiHeadResNet9(3, 10), _clients(4), ShiftTrainer(), SimilarityGrouping(4, 2, seed=0))
    server.run_round(1)
    assert all(m.local_head is not None for m in server.memory)
    server.run_round(2)  # same direction twice: stable, so the first similarity grouping happens now
    assert server.grouping.has_grouped
    assert all(m.local_head is None for m in server.memory)


def test_fedhep_end_to_end_learns_and_writes_results(tmp_path):
    cfg = load_config(overrides={
        "name": "smoke_hep", "method": "fedhep", "model": "resnet9_multihead", "data.dataset": "synthetic",
        "data.num_clients": 4, "data.train_subset": 400, "data.test_subset": 200, "data.alpha": 0.5,
        "train.local_epochs": 1, "run.rounds": 3, "run.eval_every": 1, "run.device": "cpu",
        "run.output_dir": str(tmp_path)})
    result = run_experiment(cfg)
    assert [h["round"] for h in result["history"]] == [1, 2, 3]
    assert result["final"]["pooled"] > 25.0  # 10 classes: chance is 10%
    assert (tmp_path / "smoke_hep_s42" / "metrics.json").exists()
