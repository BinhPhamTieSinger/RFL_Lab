"""The benchmark sweeps (scripts/make_benchmark.py) and how scripts/plot_benchmark.py matches runs to cells."""
import dataclasses
import importlib.util
from pathlib import Path

import pytest
import yaml

from rfl.experiment.sweep import expand

ROOT = Path(__file__).resolve().parents[1]
SWEEPS = ROOT / "configs" / "sweeps"


def _module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _runs(path):
    return dict(expand(yaml.safe_load(path.read_text())))


@pytest.fixture(scope="module")
def bench():
    return {f.name: _runs(f) for f in sorted(SWEEPS.glob("bench_*.yaml"))}


def test_every_setting_has_a_file_and_names_are_unique(bench):
    mb = _module("make_benchmark")
    assert len(bench) == len(mb.SETTINGS) + 1
    names = [n for runs in bench.values() for n in runs]
    assert len(names) == len(set(names))


def test_fedavg_family_runs_no_server_attacks(bench):
    kinds = {cfg.attack.kind for cfg in bench["bench_0_fedavg_family.yaml"].values()}
    assert kinds == {None, "sign_flip", "label_flip", "random_noise", "backdoor"}


@pytest.mark.parametrize("n", [15, 30, 50])
def test_runs_named_like_the_grid_are_the_grid_runs(bench, n):
    """Reusing a grid run by name is only valid if it is the same experiment."""
    grid = _runs(SWEEPS / f"gpu_grid_n{n}.yaml")
    shared = 0
    for runs in bench.values():
        for name, cfg in runs.items():
            if name in grid:
                assert dataclasses.asdict(cfg) == dataclasses.asdict(grid[name]), name
                shared += 1
    assert shared > 0


def test_cell_key_ignores_paths_and_evaluation_switches(bench):
    pb = _module("plot_benchmark")
    cfg = next(iter(bench["bench_1_gg15_a05.yaml"].values()))
    a = dataclasses.asdict(cfg)
    b = dataclasses.asdict(dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, root="elsewhere"),
                                               fedhep=dataclasses.replace(cfg.fedhep, exit_guard=False)))
    assert pb.cell_key(a) == pb.cell_key(b)
    c = dataclasses.asdict(dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, alpha=0.1)))
    assert pb.cell_key(a) != pb.cell_key(c)


def test_every_cell_key_is_distinct(bench):
    """Two different benchmark models must never collapse onto one cell."""
    pb = _module("plot_benchmark")
    cells = pb.cells()
    assert len(cells) == sum(len(r) for r in bench.values())


def test_first_sweeps_are_benchmark_runs(bench):
    """gpu_c8_gates.yaml runs benchmark cells early: every run must be the benchmark run of the same name."""
    mb = _module("make_benchmark")
    every = {n: cfg for runs in bench.values() for n, cfg in runs.items()}
    for fname, models in mb.FIRST.items():
        runs = _runs(SWEEPS / fname)
        assert len(runs) == 16 * len(models)   # names; each runs 3 seeds
        for name, cfg in runs.items():
            assert dataclasses.asdict(cfg) == dataclasses.asdict(every[name]), name
