#!/usr/bin/env bash
# Runs every quality gate. All must pass before an experiment result is trusted.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=${PYTHON:-python}  # the venv interpreter; tools run through it so versions match pyproject.toml
echo "== ruff (style and common bugs)";        "$PY" -m ruff check src tests
echo "== mypy (types)";                         "$PY" -m mypy src
echo "== import-linter (architecture layers)";  lint-imports
echo "== pytest + hypothesis (behaviour)";      "$PY" -m pytest -q
echo "== graphify (code graph -> graphify-out/)"; graphify update . --force >/dev/null && sed -n '/## Summary/,/## Community Hubs/p' graphify-out/GRAPH_REPORT.md | head -6
echo "All checks passed."
