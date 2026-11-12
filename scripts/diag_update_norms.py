"""Diagnostic behind the EXPERIMENT_LOG finding "BatchNorm batch counters dominate FedHEP's grouping signal".

Trains three synthetic clients with disjoint classes for one FedHEP round from the same start, then prints the norm
of each part of the flattened update and the pairwise cosines with and without the counters. CPU, a few minutes.
  python scripts/diag_update_norms.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rfl.client.fedhep import FedHEPTrainer, HeadMemory  # noqa: E402
from rfl.client.sgd import SGDSettings  # noqa: E402
from rfl.core.state import flatten  # noqa: E402
from rfl.core.types import ClientData  # noqa: E402
from rfl.experiment.runner import synthetic_data  # noqa: E402
from rfl.models.resnet9 import MultiHeadResNet9  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    x, y, _, _ = synthetic_data(3000, 10, 10, 0, torch.device("cpu"))
    y = (y * 10 + torch.randint(0, 10, y.shape)) % 100  # spread over 100 labels
    start = MultiHeadResNet9(3, 100).state_dict()
    def part(k: str) -> str:
        return "counter" if k.endswith("num_batches_tracked") else "bnstat" if "running" in k else "param"

    kind = np.concatenate([np.full(v.numel(), part(k)) for k, v in start.items()])
    rng = np.random.RandomState(1)
    deltas = []
    for c in range(3):
        idx = torch.nonzero(torch.isin(y, torch.as_tensor(rng.choice(100, 8, replace=False)))).flatten()
        model = MultiHeadResNet9(3, 100)
        model.load_state_dict(start)
        FedHEPTrainer(SGDSettings(local_epochs=5), 25, 100, 3).train(
            model, ClientData(x[idx], y[idx], x[:1], y[:1]), 1, HeadMemory())
        deltas.append((flatten(model.state_dict()) - flatten(start)).numpy())
        norms = {t: round(float(np.linalg.norm(deltas[-1][kind == t])), 1) for t in ("param", "bnstat", "counter")}
        print(f"client {c}: {len(idx)} images, update norm by part {norms}", flush=True)

    def cos(a: np.ndarray, b: np.ndarray) -> float:
        return round(float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)), 3)

    for name, sel in (("full vector (authors)", kind != ""), ("without counters", kind != "counter"),
                      ("weights only", kind == "param")):
        print(f"cosine, {name}: {[cos(deltas[i][sel], deltas[j][sel]) for i, j in ((0, 1), (0, 2), (1, 2))]}")


if __name__ == "__main__":
    main()
