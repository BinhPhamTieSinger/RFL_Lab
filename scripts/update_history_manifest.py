"""Add every finished run (a run folder whose metrics.json has final results) to the visualizer's Experiment
history (visualizer/frontend/research/history-manifest.json). Existing entries are kept.

  python3 scripts/update_history_manifest.py
"""
import glob
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
path = Path("visualizer/frontend/research/history-manifest.json")
listed = {e["source"] for e in json.loads(path.read_text())}
done = {os.path.dirname(m) for m in glob.glob("outputs*/*/metrics.json")
        if not m.startswith(("outputs_smoke", "outputs_bench")) and json.load(open(m)).get("final")}
entries = [{"source": s} for s in sorted(listed | done)]
path.write_text(json.dumps(entries, indent=2) + "\n")
print(f"{len(done - listed)} runs added to the visualizer history ({len(entries)} listed)")
