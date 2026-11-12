"""Summarise finished runs into one table (mean ± sd over seeds).

  python scripts/summarize.py outputs                # prints a markdown table
  python scripts/summarize.py outputs --csv results.csv
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rfl.experiment.summary import summarize, to_markdown  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir")
    ap.add_argument("--csv")
    ap.add_argument("--filter", default="", help="only runs whose name contains this text")
    a = ap.parse_args()
    table = [r for r in summarize(a.output_dir) if a.filter in r["name"]]
    print(to_markdown(table))
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)


if __name__ == "__main__":
    main()
