"""Run a benchmark sweep (see src/rfl/experiment/sweep.py for the spec). Finished runs are skipped, so an
interrupted sweep resumes where it stopped.

  python scripts/sweep.py configs/sweeps/phase2_baseline.yaml --dry-run    # list what would run
  python scripts/sweep.py configs/sweeps/phase2_baseline.yaml
  python scripts/sweep.py configs/sweeps/x.yaml --set data.root=data       # override every run
  python scripts/sweep.py configs/sweeps/x.yaml --shard 0/4                # runs 0, 4, 8, ... of the grid
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rfl.experiment.sweep import run_sweep  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides for every run, e.g. data.root=data")
    ap.add_argument("--shard", default="0/1", help="i/n: this process runs every n-th run, starting at i")
    a = ap.parse_args()
    extra = {k: yaml.safe_load(v) for k, v in (item.split("=", 1) for item in a.set)}
    i, n = (int(x) for x in a.shard.split("/"))
    todo = run_sweep(yaml.safe_load(Path(a.spec).read_text()), dry_run=a.dry_run, extra=extra, shard=(i, n))
    print(f"{len(todo)} run(s) {'to do' if a.dry_run else 'done'}:")
    for t in todo:
        print("  " + t)


if __name__ == "__main__":
    main()
