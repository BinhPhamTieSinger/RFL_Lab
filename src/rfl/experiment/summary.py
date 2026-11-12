"""Collects finished runs (outputs/<name>_s<seed>/metrics.json) into one row per run name: mean and standard
deviation over seeds of final accuracy, bottom-10% and attack success rate, and detection quality averaged over
rounds (then over seeds). Missing values (e.g. no ASR without a backdoor) stay empty."""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


def _mean_sd(xs: list[float]) -> tuple[float | None, float | None]:
    if not xs:
        return None, None
    return statistics.fmean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def _round_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return statistics.fmean(vals) if vals else None


def summarize(output_dir: str | Path) -> list[dict[str, Any]]:
    runs: dict[str, list[dict[str, Any]]] = {}
    for f in sorted(Path(output_dir).glob("*/metrics.json")):
        m = json.loads(f.read_text())
        runs.setdefault(m["name"], []).append(m)
    table = []
    for name, ms in runs.items():
        row: dict[str, Any] = {"name": name, "method": ms[0]["method"], "seeds": sorted(m["seed"] for m in ms),
                               "attackers": [len(m.get("attackers", [])) for m in ms]}
        for key, src in (("acc", "pooled"), ("bottom10", "bottom10")):
            row[key], row[key + "_sd"] = _mean_sd([m["final"][src] for m in ms])
        asr = [m["history"][-1]["asr"] for m in ms if m["history"][-1].get("asr") is not None]
        row["asr"], row["asr_sd"] = _mean_sd(asr)
        for key in ("precision", "recall", "f1", "auroc"):
            per_seed = [v for m in ms if (v := _round_mean(m.get("detection", []), key)) is not None]
            row[key] = statistics.fmean(per_seed) if per_seed else None
        table.append(row)
    return table


def to_markdown(table: list[dict[str, Any]]) -> str:
    def cell(row: dict[str, Any], key: str) -> str:
        v, sd = row.get(key), row.get(key + "_sd")
        if v is None:
            return "–"
        return f"{v:.2f} ± {sd:.2f}" if sd is not None else f"{v:.2f}"

    lines = ["| Run | Seeds | Attackers | Accuracy | Bottom-10% | ASR | Precision | Recall | F1 | AUROC |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    keys = ("acc", "bottom10", "asr", "precision", "recall", "f1", "auroc")
    for r in table:
        lines.append(f"| {r['name']} | {len(r['seeds'])} | {', '.join(map(str, r['attackers']))} | "
                     + " | ".join(cell(r, k) for k in keys) + " |")
    return "\n".join(lines)
