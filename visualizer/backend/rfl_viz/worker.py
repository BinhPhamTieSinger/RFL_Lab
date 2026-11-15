import json
import math
import sys
import time
import traceback
from pathlib import Path

import torch

from rfl.experiment.runner import run_experiment
from rfl_viz.observe import observed

start = time.perf_counter()


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    return value


def emit(data):
    print(
        "@rfl " + json.dumps(finite_json({"elapsed_s": time.perf_counter() - start, **data}), allow_nan=False),
        flush=True,
    )


def main():
    folder = Path(sys.argv[1])
    raw = json.loads((folder / "request.json").read_text())
    from rfl_viz.catalog import validate

    cfg = validate(raw["method"], raw["overrides"])
    from dataclasses import replace

    cfg = replace(cfg, name="result", run=replace(cfg.run, output_dir=str(folder)))
    torch.set_num_threads(2)
    try:
        emit({"type": "experiment_started"})
        with observed(emit):
            result = run_experiment(cfg)
        (folder / "result.json").write_text(json.dumps(finite_json(result)))
        emit({"type": "experiment_completed", "final": result["final"]})
    except BaseException as error:
        emit({"type": "experiment_failed", "error": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
