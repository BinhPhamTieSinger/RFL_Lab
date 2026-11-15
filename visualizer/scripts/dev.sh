#!/usr/bin/env bash
set -euo pipefail
VIZ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESEARCH_ROOT="$(cd "$VIZ_ROOT/../.." && pwd)"
PYTHON_BIN="${RFL_PYTHON:-$RESEARCH_ROOT/.venv/bin/python}"
export PYTHONPATH="$VIZ_ROOT/backend:$VIZ_ROOT/../src"
"$PYTHON_BIN" -m uvicorn rfl_viz.api:app --host 127.0.0.1 --port 8000 &
API_PID=$!
trap 'kill "$API_PID" 2>/dev/null || true' EXIT INT TERM
cd "$VIZ_ROOT/frontend"
npm run dev
