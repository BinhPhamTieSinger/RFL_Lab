"""Acer c14 and c15: the high-side gate (`plus.high_gate: 20`, c14) and the low-side gate of 6 with it
(`plus.gap_gate: 6`, c15) against FedHEP-Guard's union screen (report 8.2, P33-P36, P41-P43).

  python scripts/plot_c14.py outputs

Runs: acer_c14_high_gate.yaml (x14_*__r3wc_hg20), acer_c15_low_gate.yaml (x15_*__r3wc_g6h20). Controls, all on the Acer: x8_f3__*__r3wc, x10_a05__*__r3wc_shp,
x11_a05__minmax3__r3wc, x13_n50a01__clean__r3wc, and acer_c14_control.yaml (x14_*__r3wc) once run. For each run
the low-gate rule of report 8.2 (gap gate 6 on the low side, 20 on the high side) is also replayed on its logs.
Writes docs/figures/acer_gates.png and docs/results/acer_c14.json (acer_c14_high_gate.png, the c14-only version of
2026-11-26, is kept as it was)."""
from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pr = _module("plot_results")
rs = _module("replay_screen")
plt = pr.plt
SEEDS = (42, 123, 7)
G6H20 = {"low_gate": 6.0, "two_sided": False, "high_gate": 20.0}
# label -> (high-gate runs, union-screen control runs)
CONDITIONS = [("15 clients\nno attack", ["x14_a05__clean__r3wc_hg20"], ["x8_f3__clean__r3wc", "x10_a05__clean__r3wc_shp"]),
              ("15 clients\nALIE", ["x14_a05__alie3__r3wc_hg20"], ["x8_f3__alie3__r3wc"]),
              ("15 clients\nMin-Max", ["x14_a05__minmax3__r3wc_hg20"], ["x11_a05__minmax3__r3wc"]),
              ("15 clients\nbackdoor", ["x14_a05__backdoor3__r3wc_hg20"],
               ["x8_f3__backdoor3__r3wc", "x10_a05__backdoor3__r3wc_shp"]),
              ("15 clients\nstealth backdoor", ["x14_a05__stealth3__r3wc_hg20"], ["x14_a05__stealth3__r3wc"]),
              ("50 clients, α 0.1\nno attack", ["x14_n50a01__clean__r3wc_hg20"], ["x13_n50a01__clean__r3wc"]),
              ("50 clients, α 0.1\nbackdoor", ["x14_n50a01__backdoor10__r3wc_hg20"], ["x14_n50a01__backdoor10__r3wc"])]


def run(m: dict) -> dict:
    bad = set(m["attackers"])
    first: dict[int, int] = {}
    for d in m["diagnostics"]:
        for c in d["isolated"]:
            first.setdefault(c, d["round"])
    low = rs.replay(m, **G6H20)
    return {"seed": m["seed"], "acc": m["final"]["pooled"], "asr": m["history"][-1].get("asr") or 0.0,
            "attackers": len(bad), "caught": sum(c in bad for c in first), "honest": sum(c not in bad for c in first),
            "rounds": sorted(r for c, r in first.items() if c in bad),
            "replay_low_gate": {"caught": sum(c in bad for c in low), "honest": sum(c not in bad for c in low),
                                "rounds": sorted(r for c, r in low.items() if c in bad)}}


def cell(outputs: Path, names: list[str]) -> list[dict]:
    out = []
    for name in names:
        for seed in SEEDS:
            p = outputs / f"{name}_s{seed}" / "metrics.json"
            if p.exists():
                m = json.loads(p.read_text())
                if m.get("device") == "xpu" and m.get("final"):
                    out.append(run(m))
    return out


def fig(res: dict) -> None:
    labels = list(res)
    arms = (("control", "union screen (FedHEP-Guard's screen)", pr.SLOT[0]),
            ("high_gate", "high side only at gap ratio ≥ 20 (c14)", pr.SLOT[1]),
            ("low_gate", "and low side at gap ratio ≥ 6 (c15)", pr.SLOT[2]))
    w = 0.8 / len(arms)
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.6), gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    for j, (arm, name, color) in enumerate(arms):
        xs = [i + (j - 1) * w for i in range(len(labels))]
        for x, lab in zip(xs, labels, strict=True):
            rs_ = res[lab][arm]
            if not rs_:
                ax.text(x, 0.2, "not run", ha="center", fontsize=6.5, color=pr.INK2)
                continue
            h = statistics.fmean(r["honest"] for r in rs_)
            ax.bar(x, h, w * 0.95, color=color, label=name if lab == labels[0] else None)
            ax.text(x, h + 0.2, f"{h:.1f}", ha="center", fontsize=7.5)
            ax.text(x, -1.1, f"n={len(rs_)}", ha="center", fontsize=6, color=pr.INK2)
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    ax.set_ylim(-1.5, None)
    ax.set_title("honest clients isolated per run (Acer, 30 rounds)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax = axes[1]
    bd = [lab for lab in labels if "backdoor" in lab]
    for j, (arm, _name, color) in enumerate(arms):
        xs = [i + (j - 1) * w for i in range(len(bd))]
        for x, lab in zip(xs, bd, strict=True):
            rs_ = res[lab][arm]
            if not rs_:
                ax.text(x, 2, "not run", ha="center", fontsize=6.5, color=pr.INK2)
                continue
            ax.bar(x, statistics.fmean(r["asr"] for r in rs_), w * 0.95, color=color, alpha=0.85)
            for r in rs_:
                ax.scatter(x, r["asr"], color=pr.INK, s=12, zorder=3)
                if r["asr"] > 5:
                    ax.annotate(f"seed {r['seed']}", (x, r["asr"]), xytext=(4, 0), textcoords="offset points",
                                fontsize=6.5, va="center")
    ax.set_xticks(range(len(bd)), bd, fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_title("backdoor success (%), dots = seeds", fontsize=10)
    fig.suptitle("Acer c14 and c15: the high-side gate alone halves honest isolations at 50 clients but let the "
                 "stealth backdoor through on 2 seeds; adding the low-side gate of 6 catches it on every seed, but "
                 "honest isolations come back", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    pr.save(fig, "acer_gates.png")


def main() -> None:
    pr.style()
    outputs = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs")
    res = {lab: {"control": cell(outputs, ctrl), "high_gate": cell(outputs, hg),
                 "low_gate": cell(outputs, [n.replace("x14_", "x15_").replace("_hg20", "_g6h20") for n in hg])}
           for lab, hg, ctrl in CONDITIONS}
    (ROOT / "docs" / "results" / "acer_c14.json").write_text(json.dumps(res, indent=1))
    fig(res)
    for lab, arms in res.items():
        print(lab.replace("\n", " "))
        for arm, rs_ in arms.items():
            if rs_:
                print(f"  {arm:9s} n={len(rs_)} honest {statistics.fmean(r['honest'] for r in rs_):.1f} "
                      f"caught {sum(r['caught'] for r in rs_)}/{sum(r['attackers'] for r in rs_)} "
                      f"asr {[round(r['asr'], 1) for r in rs_]} acc {statistics.fmean(r['acc'] for r in rs_):.1f} "
                      f"rounds {[r['rounds'] for r in rs_]} | low gate replay "
                      f"{[(r['replay_low_gate']['caught'], r['replay_low_gate']['rounds'][:1]) for r in rs_]}")


if __name__ == "__main__":
    main()
