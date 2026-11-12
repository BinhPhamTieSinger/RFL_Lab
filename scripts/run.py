"""Run an experiment preset for one or more seeds.

  python scripts/run.py configs/cifar100_fedavg.yaml --seeds 42 123 7
  python scripts/run.py configs/cifar100_fedavg.yaml --set data.alpha=0.05 name=cifar100_fedavg_a0.05
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rfl.experiment.config import load_config  # noqa: E402
from rfl.experiment.runner import run_seeds  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. data.alpha=0.05")
    a = ap.parse_args()
    overrides = {k: yaml.safe_load(v) for k, v in (item.split("=", 1) for item in a.set)}
    results = run_seeds(load_config(a.config, overrides), a.seeds)
    accs = [r["final"]["pooled"] for r in results]
    print(f"final accuracy per seed: {[round(x, 2) for x in accs]}  mean: {sum(accs) / len(accs):.2f}")


if __name__ == "__main__":
    main()
