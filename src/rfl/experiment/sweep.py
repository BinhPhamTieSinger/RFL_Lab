"""Benchmark sweeps: every combination of a few labelled axes, for every seed, skipping runs already done.

Spec (YAML):
  seeds: [42, 123, 7]
  common: {data.alpha: 0.5, run.rounds: 30}          # applied to every run
  axes:                                                # one entry per axis: label -> overrides
    - fedavg: {config: configs/cifar100_fedavg.yaml}  # "config" picks the preset
      defended: {config: configs/cifar100_fedhep_defended.yaml}
    - clean: {}
      sign3: {attack.kind: sign_flip, attack.count: 3}
Run names join the labels: "fedavg__clean", "defended__sign3", ... Later axes override earlier ones, and every
axis overrides `common`; `extra` (the command line's --set) overrides everything.

Sharding: `shard=(i, n)` keeps every n-th (run, seed) pair starting at i, counted over the whole grid, so n
processes with i = 0..n-1 split one sweep without overlap and each still skips what is already done."""
from __future__ import annotations

import itertools
from dataclasses import replace
from pathlib import Path
from typing import Any

from rfl.experiment.config import ExperimentConfig, load_config
from rfl.experiment.runner import run_experiment


def expand(spec: dict[str, Any], extra: dict[str, Any] | None = None) -> list[tuple[str, ExperimentConfig]]:
    axes: list[dict[str, dict[str, Any]]] = spec.get("axes", [])
    runs = []
    for combo in itertools.product(*(list(axis.items()) for axis in axes)):
        overrides = dict(spec.get("common", {}))
        config_path = None
        for _, over in combo:
            over = dict(over)
            config_path = over.pop("config", config_path)
            overrides.update(over)
        overrides.update(extra or {})
        name = "__".join(label for label, _ in combo)
        overrides["name"] = name
        runs.append((name, load_config(config_path, overrides)))
    return runs


def run_sweep(spec: dict[str, Any], dry_run: bool = False, extra: dict[str, Any] | None = None,
              shard: tuple[int, int] = (0, 1)) -> list[str]:
    """Runs what is missing in this shard; returns the run folders it ran (or would run, with dry_run)."""
    todo = []
    pairs = [(cfg, seed) for _, cfg in expand(spec, extra) for seed in spec.get("seeds", [cfg.run.seed])]
    for index, (cfg, seed) in enumerate(pairs):
        if index % shard[1] != shard[0]:
            continue
        c = replace(cfg, run=replace(cfg.run, seed=seed))
        folder = Path(c.run.output_dir) / f"{c.name}_s{seed}"
        if (folder / "metrics.json").exists():
            continue
        todo.append(str(folder))
        if not dry_run:
            run_experiment(c)
    return todo
