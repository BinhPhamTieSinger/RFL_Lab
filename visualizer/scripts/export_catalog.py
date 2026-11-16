"""Regenerate frontend/research/catalog.json, the static copy of the API's catalog that the deployed site uses:
method list, configuration schemas, formula registry, documentation and the source files the formulas link to.
Run after changing the research implementation, the schema or the formulas:

  PYTHONPATH=visualizer/backend:src python visualizer/scripts/export_catalog.py
"""
import json
from pathlib import Path

from rfl_viz.catalog import FORMULAS, LAB, METHODS, schema

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sources = {}
    for f in FORMULAS:
        sources[f["source"]] = (LAB / "src/rfl" / f["source"]).read_text()
    catalog = {
        "models": [{"id": m, "hierarchical": m.startswith("fedhep")} for m in METHODS],
        "schemas": {m: schema(m) for m in METHODS},
        "formulas": FORMULAS,
        "documentation": (ROOT / "README.md").read_text(),
        "sources": sources,
    }
    out = ROOT / "frontend/research/catalog.json"
    out.write_text(json.dumps(catalog, indent=2) + "\n")
    print(f"wrote {out.relative_to(ROOT.parent)}: {len(FORMULAS)} formulas, {len(sources)} sources")


if __name__ == "__main__":
    main()
