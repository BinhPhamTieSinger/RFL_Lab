"""Spike (Stage 1): observe a running FedHEP-family server without changing what it computes.

`observed(emit)` is a context manager that wraps the method builders in rfl.experiment.registry.METHODS, so the
server `run_experiment` builds gets instance-level wrappers on the components it calls each round. Wrappers only
read arguments and results, measure wall time, and call `emit(event)`. They draw no random numbers and write nothing
back, so a run under observation must produce exactly the same metrics as the same run without it (tested).
"""
from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterator
from typing import Any

import torch

from rfl.experiment import registry

Emit = Callable[[dict[str, Any]], None]


def _wrap(obj: Any, name: str, before: Callable[..., None] | None = None,
          after: Callable[..., None] | None = None) -> None:
    """Replace obj.name by a wrapper on this instance only. `before(*args)`, `after(result, seconds, *args)`."""
    original = getattr(obj, name)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if before:
            before(*args, **kwargs)
        t0 = time.perf_counter()
        result = original(*args, **kwargs)
        if after:
            after(result, time.perf_counter() - t0, *args, **kwargs)
        return result

    setattr(obj, name, wrapper)


def instrument_fedhep(server: Any, emit: Emit) -> None:
    """FedHEP family: client training, attack hooks, regrouping, per-parent and Global aggregation."""
    state: dict[str, Any] = {"round": 0, "client": None, "group": None}
    memory_ids = {id(m): i for i, m in enumerate(server.memory)}

    def round_start(round_idx: int) -> None:
        state["round"] = round_idx
        emit({"type": "round_started", "round": round_idx})

    def round_end(_: Any, seconds: float, round_idx: int) -> None:
        emit({"type": "round_completed", "round": round_idx, "seconds": seconds})

    _wrap(server, "run_round", round_start, round_end)

    def train_start(model: Any, data: Any, round_idx: int, memory: Any, targets: Any = None) -> None:
        state["client"] = memory_ids[id(memory)]
        emit({"type": "client_training_started", "round": round_idx, "client": state["client"],
              "samples": int(data.num_train), "poisoned_labels": targets is not None})

    def train_end(_: Any, seconds: float, model: Any, data: Any, round_idx: int, memory: Any,
                  targets: Any = None) -> None:
        emit({"type": "client_training_completed", "round": round_idx, "client": state["client"],
              "seconds": seconds,
              "r_skew": memory.r_skew, "head_losses": dict(memory.head_losses),
              "ensemble_alpha": list(memory.ensemble_alpha)})

    _wrap(server.trainer, "train", train_start, train_end)

    if server.attack is not None:
        def upload_end(result: torch.Tensor, seconds: float, trained: torch.Tensor, start: torch.Tensor) -> None:
            emit({"type": "attack_executed", "round": state["round"], "client": state["client"],
                  "attack": type(server.attack).__name__,
                  "clean_update_norm": float((trained - start).norm()),
                  "sent_update_norm": float((result - start).norm()),
                  "cosine_clean_vs_sent": float(torch.nn.functional.cosine_similarity(
                      trained - start, result - start, dim=0))})
        _wrap(server.attack, "poison_upload", after=upload_end)

    def members_end(result: list[int], seconds: float, group: int, among: Any) -> None:
        state["group"] = (group, list(result))

    _wrap(server.grouping, "members", after=members_end)

    def regroup_end(event: Any, seconds: float, deltas: Any, round_idx: int) -> None:
        emit({"type": "grouping_step", "round": round_idx, "event": event.value, "seconds": seconds,
              "assignment": dict(server.grouping.assignment)})

    _wrap(server.grouping, "step", after=regroup_end)

    for tier in ("parent_defense", "global_defense"):
        if hasattr(server, tier):
            def agg_end(result: Any, seconds: float, updates: Any, masks: Any, _tier: str = tier) -> None:
                _, trust = result
                group = state["group"] if _tier == "parent_defense" else None
                emit({"type": "aggregation_completed", "round": state["round"], "tier": _tier,
                      "group": group[0] if group else None, "members": group[1] if group else None,
                      "senders": len(updates), "trust": [float(t) for t in trust.tolist()], "seconds": seconds})
            _wrap(getattr(server, tier), "aggregate", after=agg_end)


@contextlib.contextmanager
def observed(emit: Emit) -> Iterator[None]:
    """Within this block, every server built through METHODS is instrumented."""
    saved = dict(registry.METHODS)

    def wrap_builder(build: Callable[..., Any]) -> Callable[..., Any]:
        def builder(cfg: Any, model: Any, clients: Any) -> Any:
            server = build(cfg, model, clients)
            if hasattr(server, "memory") and hasattr(server, "grouping"):
                instrument_fedhep(server, emit)
            return server
        return builder

    try:
        for name, build in saved.items():
            registry.METHODS[name] = wrap_builder(build)
        yield
    finally:
        registry.METHODS.clear()
        registry.METHODS.update(saved)
