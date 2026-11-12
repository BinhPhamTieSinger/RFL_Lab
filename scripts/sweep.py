"""Run a benchmark sweep (see src/rfl/experiment/sweep.py for the spec). Finished runs are skipped, so an
interrupted sweep resumes where it stopped.

  python scripts/sweep.py configs/sweeps/phase2_baseline.yaml --dry-run    # list what would run
  python scripts/sweep.py configs/sweeps/phase2_baseline.yaml
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
    a = ap.parse_args()
    todo = run_sweep(yaml.safe_load(Path(a.spec).read_text()), dry_run=a.dry_run)
    print(f"{len(todo)} run(s) {'to do' if a.dry_run else 'done'}:")
    for t in todo:
        print("  " + t)


if __name__ == "__main__":
    main()
