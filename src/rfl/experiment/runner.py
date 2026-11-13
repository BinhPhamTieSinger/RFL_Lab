"""Runs one experiment for one seed and writes `config.yaml` and `metrics.json` to its own folder."""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from rfl.core.types import ClientData
from rfl.data.cifar100 import load_cifar100
from rfl.data.clients import build_clients
from rfl.data.partition import make_partitioner
from rfl.evaluation.accuracy import evaluate_clients
from rfl.evaluation.backdoor import attack_success_rate
from rfl.evaluation.detection import detection_metrics
from rfl.experiment.config import ExperimentConfig, to_dict
from rfl.experiment.registry import METHODS, MODELS


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():  # Intel GPUs (e.g. the Acer's Arc 130V)
        return torch.device("xpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Same RNGs as the authors' `set_seed`."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def synthetic_data(n_train: int, n_test: int, num_classes: int, seed: int, device: torch.device
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Smooth class patterns plus noise: learnable in a few rounds. For tests only."""
    g = torch.Generator().manual_seed(seed)
    protos = torch.nn.functional.interpolate(torch.randn(num_classes, 3, 4, 4, generator=g), size=(32, 32),
                                             mode="bilinear", align_corners=False)

    def draw(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        y = torch.randint(0, num_classes, (n,), generator=g)
        return (protos[y] + 0.5 * torch.randn(n, 3, 32, 32, generator=g)).to(device), y.to(device)

    xtr, ytr = draw(n_train)
    xte, yte = draw(n_test)
    return xtr, ytr, xte, yte


def run_experiment(cfg: ExperimentConfig) -> dict[str, Any]:
    out = Path(cfg.run.output_dir) / f"{cfg.name}_s{cfg.run.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(to_dict(cfg), sort_keys=False))
    device = pick_device(cfg.run.device)
    set_seed(cfg.run.seed)

    d = cfg.data
    if d.dataset == "cifar100":
        x_tr, y_tr, x_te, y_te = load_cifar100(d.root, d.train_subset, d.test_subset, cfg.run.seed, device)
        num_classes = 100
    elif d.dataset == "synthetic":
        num_classes = 10
        x_tr, y_tr, x_te, y_te = synthetic_data(d.train_subset or 600, d.test_subset or 200, num_classes,
                                                cfg.run.seed, device)
    else:
        raise ValueError(f"Unknown dataset {d.dataset!r}")
    clients: list[ClientData] = build_clients(x_tr, y_tr, x_te, y_te, make_partitioner(d.alpha), d.num_clients,
                                              cfg.run.seed)

    model = MODELS[cfg.model](3, num_classes).to(device)
    server = METHODS[cfg.method](cfg, model, clients)
    attackers = frozenset(getattr(server, "attackers", frozenset()))
    attack = getattr(server, "attack", None)
    trigger = getattr(attack, "trigger", None)
    history: list[dict[str, Any]] = []
    detection: list[dict[str, Any]] = []
    t0 = time.time()
    for r in range(1, cfg.run.rounds + 1):
        server.run_round(r)
        report = getattr(server, "report", None)
        if report is not None:
            det = {"round": r, **detection_metrics(report, attackers, len(clients))}
            diagnostics = getattr(server, "diagnostics", [])
            if diagnostics and "isolated" in diagnostics[-1]:
                isolated = set(diagnostics[-1]["isolated"])
                honest_n = len(clients) - len(attackers)
                det["false_isolation_rate"] = len(isolated - attackers) / honest_n if honest_n else None
                det["isolated"] = sorted(isolated)
            detection.append(det)
        if r % cfg.run.eval_every == 0 or r == cfg.run.rounds:
            rep = evaluate_clients(server.predict, clients, exclude=getattr(server, "unscored", frozenset()))
            row: dict[str, Any] = {"round": r, "pooled_acc": rep.pooled, "mean_client_acc": rep.mean_client,
                                   "bottom10": rep.bottom10, "elapsed_s": time.time() - t0}
            if trigger is not None:
                row["asr"] = attack_success_rate(server.predict, clients, trigger, cfg.attack.backdoor_target,
                                                 exclude=attackers)
            history.append(row)
            asr = f"  asr={row['asr']:.2f}" if row.get("asr") is not None else ""
            print(f"[{cfg.name} s{cfg.run.seed}] round {r}/{cfg.run.rounds}  acc={rep.pooled:.2f}  "
                  f"bottom10={rep.bottom10:.2f}{asr}", flush=True)
    result = {"name": cfg.name, "method": cfg.method, "seed": cfg.run.seed, "alpha": d.alpha,
              "final": asdict(rep), "history": history, "device": str(device),
              "evaluation_scope": "all honest clients with test data, including isolated clients",
              "client_label_stats": [{"client": i, "classes": int(c.y_train.unique().numel()),
                  "label_entropy": float(-(lambda p: p * p.clamp_min(1e-12).log())(
                      torch.bincount(c.y_train, minlength=num_classes).float() / max(1, c.num_train)).sum())}
                  for i, c in enumerate(clients)],
              "client_train_sizes": [c.num_train for c in clients],
              "attackers": sorted(attackers), "detection": detection,
              "model_floats": sum(v.numel() for v in model.state_dict().values()),
              "diagnostics": getattr(server, "diagnostics", None)}
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def run_seeds(cfg: ExperimentConfig, seeds: list[int]) -> list[dict[str, Any]]:
    return [run_experiment(replace(cfg, run=replace(cfg.run, seed=s))) for s in seeds]
