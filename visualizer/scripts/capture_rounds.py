"""Capture controlled, real CIFAR-100 visualization rounds outside curated research history."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

VIZ = Path(__file__).resolve().parents[1]
LAB = VIZ.parent
OUT = VIZ / "frontend/research/captures"
OUT.mkdir(parents=True, exist_ok=True)
methods = sys.argv[1:] or [
    "fedavg",
    "fedprox",
    "scaffold",
    "multikrum",
    "ditto",
    "fedrep",
    "fedhep",
    "fedhep_defended",
    "fedhep_plus",
    "fedhep_robust",
]
for method in methods:
    variants = ["sign_flip", None] if method == "fedhep_plus" else ["sign_flip"]
    for attack in variants:
        capture_id = f"{method}-{attack or 'clean'}"
        destination = OUT / f"{capture_id}.json"
        folder = VIZ / "data/captures" / capture_id
        folder.mkdir(parents=True, exist_ok=True)
        request = {
            "method": method,
            "overrides": {
                "data.dataset": "cifar100",
                "data.train_subset": 300,
                "data.test_subset": 150,
                "data.num_clients": 6,
                "data.alpha": 0.5,
                "run.rounds": 1,
                "run.device": "cpu",
                "train.local_epochs": 1,
                "attack.kind": attack,
                "attack.count": 1 if attack else 0,
            },
        }
        (folder / "request.json").write_text(json.dumps(request))
        events = []
        process = subprocess.Popen(
            [sys.executable, "-u", "-m", "rfl_viz.worker", str(folder)],
            cwd=LAB,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONPATH": f"{VIZ / 'backend'}:{LAB / 'src'}"},
        )
        for line in process.stdout:
            if line.startswith("@rfl "):
                events.append({"timestamp": time.time(), **json.loads(line[5:])})
            else:
                events.append({"type": "log", "timestamp": time.time(), "text": line.rstrip()})
        if process.wait():
            raise RuntimeError(f"{capture_id} failed. {events[-3:]}")
        events.append({"type": "worker_stopped", "status": "completed", "timestamp": time.time()})
        capture = {
            "id": capture_id,
            "config": request,
            "events": events,
            "result": json.loads((folder / "result.json").read_text()),
            "purpose": "visualization",
        }
        destination.write_text(json.dumps(capture, separators=(",", ":")))
        print(f"Captured {capture_id}: {len(events)} events", flush=True)
