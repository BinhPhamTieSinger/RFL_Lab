"""Experiment 1 of the literature review (Part 2): are honest isolations a multiple-testing accident, or do the same
honest clients stay unusual round after round? Offline, on the union screen's logged scores with no attack.

  python scripts/diag_honest_null.py outputs

For every GPU run `gg*__clean__r3wc_s*` (FedHEP-Guard's screen, no attack):
  1. persistence: lag-1 autocorrelation of each honest client's z over rounds, on both signals;
  2. a null without persistence: every round, the logged scores are dealt to the clients at random (same scores, same
     gap ratios, same gate), and the screen's rule is replayed (scripts/replay_screen.py). If isolations come from
     multiple testing alone, the shuffled logs isolate as many clients as the real ones;
  3. who is isolated: label entropy and data size of isolated against kept honest clients, as ranks within the run.
Writes docs/figures/honest_null.png and docs/results/honest_null.json."""
from __future__ import annotations

import glob
import importlib.util
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _module(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pr = _module("plot_results")
rs = _module("replay_screen")
plt = pr.plt
SHUFFLES = 20


def load(outputs: Path) -> list[tuple[str, dict]]:
    out = []
    for p in sorted(glob.glob(str(outputs / "gg*__clean__r3wc_s*" / "metrics.json"))):
        m = json.loads(Path(p).read_text())
        if m.get("device") == "cuda":
            out.append((Path(p).parent.name.split("__")[0], m))
    return out


def autocorr(rows: list[dict], key: str) -> list[float]:
    series: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        for c, v in r[key].items():
            series[c].append(v)
    out = []
    for v in series.values():
        if len(v) >= 8 and np.std(v) > 0:
            x = np.asarray(v)
            out.append(float(np.corrcoef(x[:-1], x[1:])[0, 1]))
    return out


def shuffled(rows: list[dict], rng: random.Random) -> list[dict]:
    out = []
    for r in rows:
        ids = list(r["zw"])
        perm = ids[:]
        rng.shuffle(perm)
        mp = dict(zip(ids, perm, strict=True))
        out.append({**r, "zw": {mp[c]: v for c, v in r["zw"].items()},
                    "zc": {mp[c]: v for c, v in r["zc"].items() if c in mp}})
    return out


def ranks(values: list[float]) -> list[float]:
    order = np.argsort(np.argsort(values))
    return list(order / max(1, len(values) - 1))


def main() -> None:
    pr.style()
    runs = load(Path(sys.argv[1] if len(sys.argv) > 1 else "outputs"))
    rng = random.Random(0)
    res: dict = defaultdict(lambda: {"runs": 0, "ac_w": [], "ac_c": [], "real": [], "null": [],
                                     "iso_entropy": [], "kept_entropy": [], "iso_size": [], "kept_size": []})
    for setting, m in runs:
        rows = rs.rounds(m)
        cell = res[setting]
        cell["runs"] += 1
        cell["ac_w"] += autocorr(rows, "zw")
        cell["ac_c"] += autocorr(rows, "zc")
        real = rs.replay(m, rows=rows)
        cell["real"].append(len(real))
        cell["null"].append(statistics.fmean(len(rs.replay(m, rows=shuffled(rows, rng))) for _ in range(SHUFFLES)))
        ent = [s["label_entropy"] for s in m["client_label_stats"]]
        size = m["client_train_sizes"]
        er, sr = ranks(ent), ranks(size)
        for c in range(len(ent)):
            (cell["iso_entropy"] if c in real else cell["kept_entropy"]).append(er[c])
            (cell["iso_size"] if c in real else cell["kept_size"]).append(sr[c])
    order = sorted(res, key=lambda s: (int(s[2:4]), s))
    summary = {s: {"runs": res[s]["runs"],
                   "autocorr_whole": statistics.median(res[s]["ac_w"]),
                   "autocorr_critical": statistics.median(res[s]["ac_c"]),
                   "isolated_real": statistics.fmean(res[s]["real"]),
                   "isolated_shuffled": statistics.fmean(res[s]["null"]),
                   "entropy_rank_isolated": statistics.fmean(res[s]["iso_entropy"]) if res[s]["iso_entropy"] else None,
                   "size_rank_isolated": statistics.fmean(res[s]["iso_size"]) if res[s]["iso_size"] else None}
               for s in order}
    (ROOT / "docs" / "results" / "honest_null.json").write_text(json.dumps(summary, indent=1))
    label = {s: f"{int(s[2:4])} clients\nα {s.split('_a')[1].replace('0', '0.', 1)}" for s in order}
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    ax = axes[0]
    for j, (key, name, color) in enumerate((("ac_w", "whole update", pr.SLOT[0]),
                                            ("ac_c", "critical layers", pr.SLOT[1]))):
        xs = [i + (j - 0.5) * 0.38 for i in range(len(order))]
        vals = [statistics.median(res[s][key]) for s in order]
        ax.bar(xs, vals, 0.36, color=color, label=name)
        for x, v in zip(xs, vals, strict=True):
            ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.axhline(0, color=pr.INK2, linewidth=0.8)
    ax.set_xticks(range(len(order)), [label[s] for s in order], fontsize=7.5)
    ax.set_title("persistence: median lag-1 autocorrelation of an honest\nclient's z from one round to the next",
                 fontsize=9.5)
    ax.legend(fontsize=8, loc="lower right")
    ax = axes[1]
    for j, (key, name, color) in enumerate((("real", "real logs", pr.INK),
                                            ("null", f"scores dealt at random each round ({SHUFFLES} shuffles)",
                                             pr.SLOT[2]))):
        xs = [i + (j - 0.5) * 0.38 for i in range(len(order))]
        vals = [statistics.fmean(res[s][key]) for s in order]
        ax.bar(xs, vals, 0.36, color=color, label=name)
        for x, v in zip(xs, vals, strict=True):
            ax.text(x, v + 0.2, f"{v:.1f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(order)), [label[s] for s in order], fontsize=7.5)
    ax.set_title("honest clients isolated per run, no attack:\nthe screen's rule on real against shuffled logs",
                 fontsize=9.5)
    ax.legend(fontsize=7.5)
    ax = axes[2]
    for j, (key, name, color) in enumerate((("iso_entropy", "label entropy (low = skewed)", pr.SLOT[0]),
                                            ("iso_size", "training-set size", pr.SLOT[3]))):
        xs = [i + (j - 0.5) * 0.38 for i in range(len(order))]
        vals = [statistics.fmean(res[s][key]) if res[s][key] else float("nan") for s in order]
        ax.bar(xs, vals, 0.36, color=color, label=name)
        for x, v, s in zip(xs, vals, order, strict=True):
            ax.text(x, (v if v == v else 0) + 0.01, f"{v:.2f}\nn={len(res[s][key])}", ha="center", fontsize=6.5)
    ax.axhline(0.5, color=pr.INK2, linestyle=":", linewidth=1)
    ax.text(-0.45, 0.47, "0.5 = no difference from the run's average client", fontsize=7, va="top", color=pr.INK2)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(order)), [label[s] for s in order], fontsize=7.5)
    ax.set_title("who is isolated: mean rank within the run (0 lowest, 1 highest)\nof isolated honest clients",
                 fontsize=9.5)
    ax.legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Honest isolations with no attack (GPU, FedHEP-Guard's screen): persistent offsets or multiple "
                 "testing?", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    pr.save(fig, "honest_null.png")
    for s, v in summary.items():
        print(s, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()})


if __name__ == "__main__":
    main()
