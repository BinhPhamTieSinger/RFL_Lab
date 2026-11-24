"""Phase 8 (Acer): the shared signal against the whole-update signal in FedHEP-Guard's union screen.

  python scripts/plot_phase8.py outputs

Runs: acer_c11_shared.yaml and acer_c11_control.yaml (x11_, 30 rounds), acer_c12_shared_durability.yaml (x12_, 100
rounds). Controls, all on the Acer: x10_*__r3wc_shp and x8_f3__*__r3wc (union screen, 30 rounds), x9__*__r3wc_tr
(FedHEP-Guard, 100 rounds). Writes docs/figures/phase8_shared.png, phase8_seed7.png and docs/results/phase8.json."""
from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rfl.defense.screening import robust_z  # noqa: E402

_spec = importlib.util.spec_from_file_location("plot_results", Path(__file__).with_name("plot_results.py"))
pr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pr)
_spec6 = importlib.util.spec_from_file_location("plot_phase6", Path(__file__).with_name("plot_phase6.py"))
p6 = importlib.util.module_from_spec(_spec6)
_spec6.loader.exec_module(p6)
plt = pr.plt
SEEDS = (42, 123, 7)
# condition label -> (shared runs, union-screen control runs)
CONDITIONS = [("α 0.5\nno attack", ["x11_a05__clean__r3sc"], ["x10_a05__clean__r3wc_shp", "x8_f3__clean__r3wc"]),
              ("α 0.5\nsign flip", ["x11_a05__sign3__r3sc"], ["x8_f3__sign3__r3wc"]),
              ("α 0.5\nALIE", ["x11_a05__alie3__r3sc"], ["x8_f3__alie3__r3wc"]),
              ("α 0.5\nMin-Max", ["x11_a05__minmax3__r3sc"], ["x11_a05__minmax3__r3wc"]),
              ("α 0.5\nbackdoor", ["x11_a05__backdoor3__r3sc"], ["x10_a05__backdoor3__r3wc_shp", "x8_f3__backdoor3__r3wc"]),
              ("α 0.1\nno attack", ["x11_a01__clean__r3sc"], ["x10_a01__clean__r3wc_shp"]),
              ("α 0.1\nbackdoor", ["x11_a01__backdoor3__r3sc"], ["x10_a01__backdoor3__r3wc_shp"])]
DURABILITY = [("plain backdoor\n100 rounds", "x12__backdoor3__r3sc_tr", "x9__backdoor3__r3wc_tr"),
              ("stealth backdoor\n100 rounds", "x12__stealth3__r3sc_tr", "x9__stealth3__r3wc_tr")]


def run(m: dict) -> dict:
    bad = set(m["attackers"])
    iso = set(m["diagnostics"][-1]["isolated"])
    _, grouped = p6._group_events(m)
    return {"seed": m["seed"], "acc": m["final"]["pooled"], "asr": m["history"][-1].get("asr") or 0.0,
            "caught": len(iso & bad), "attackers": len(bad), "honest": len(iso - bad), "grouped": grouped,
            "device": m["device"]}


def cell(runs: dict, names: list[str]) -> list[dict]:
    return [run(m) for n in names for m in runs.get(n, []) if m["seed"] in SEEDS]


def summarize(runs: dict) -> dict:
    out = {}
    for label, shared, union in CONDITIONS:
        out[label] = {"shared": cell(runs, shared), "union": cell(runs, union)}
    for label, shared, guard in DURABILITY:
        out[label] = {"shared": cell(runs, [shared]), "union": cell(runs, [guard])}
    return out


def fig_shared(res: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1.5, 1.5, 1]})
    labels = [c[0] for c in CONDITIONS]
    arms = (("union", "union screen (whole update ∪ critical layers)", pr.SLOT[0]),
            ("shared", "shared union screen (shared ∪ critical layers)", pr.SLOT[1]))
    ax = axes[0]
    for j, (arm, name, color) in enumerate(arms):
        xs = [i + (j - 0.5) * 0.38 for i in range(len(labels))]
        hon = [statistics.fmean(r["honest"] for r in res[lab][arm]) if res[lab][arm] else 0 for lab in labels]
        grp = [statistics.fmean(r["grouped"] for r in res[lab][arm]) if res[lab][arm] else 0 for lab in labels]
        ax.bar(xs, hon, 0.36, color=color, label=name)
        ax.bar(xs, grp, 0.36, color="none", edgecolor=pr.INK, hatch="////", linewidth=0)
        for x, h, lab in zip(xs, hon, labels, strict=True):
            ax.text(x, h + 0.1, f"{h:.1f}", ha="center", fontsize=7.5)
            ax.text(x, -0.55, f"n={len(res[lab][arm])}", ha="center", fontsize=6, color=pr.INK2)
    ax.bar([0], [0], color="none", edgecolor=pr.INK, hatch="////", label="of which in whole-group events")
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    ax.set_ylim(-0.8, None)
    ax.set_title("honest clients isolated per run (Acer, 30 rounds)", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right")
    ax = axes[1]
    bd = [lab for lab in res if "backdoor" in lab]
    for j, (arm, _name, color) in enumerate(arms):
        xs = [i + (j - 0.5) * 0.38 for i in range(len(bd))]
        for x, lab in zip(xs, bd, strict=True):
            vals = [r["asr"] for r in res[lab][arm]]
            ax.bar(x, statistics.fmean(vals) if vals else 0, 0.36, color=color, alpha=0.85)
            for r in res[lab][arm]:
                ax.scatter(x, r["asr"], color=pr.INK, s=12, zorder=3)
                if r["asr"] > 5:
                    ax.annotate(f"seed {r['seed']}", (x, r["asr"]), xytext=(4, 0), textcoords="offset points",
                                fontsize=6.5, va="center")
    ax.set_xticks(range(len(bd)), [b.replace("\nbackdoor", " backdoor\n30 rounds") for b in bd], fontsize=8)
    ax.set_title("backdoor success (%), dots = seeds; 100-round runs include the rewind", fontsize=10)
    ax = axes[2]
    for j, (arm, _name, color) in enumerate(arms):
        acc = [statistics.fmean(r["acc"] for r in res[lab][arm]) if res[lab][arm] else 0 for lab in labels]
        ax.plot(range(len(labels)), acc, marker="o", color=color)
    ax.set_xticks(range(len(labels)), [lab.replace("\n", " ") for lab in labels], rotation=45, ha="right",
                  fontsize=7.5)
    ax.set_title("accuracy of honest clients (%)", fontsize=10)
    fig.suptitle("Phase 8 (Acer): the shared signal cuts honest isolations, but loses the backdoor on seed 7",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    pr.save(fig, "phase8_shared.png")


def _z(scores) -> dict:
    keys = [int(c) for c in scores]
    return dict(zip(keys, robust_z(torch.tensor(list(scores.values()), dtype=torch.float64)).tolist(), strict=True))


def fig_seed7(runs: dict) -> None:
    """Why seed 7 escaped: attacker z and gap ratio per round, whole update against shared, rounds 1-8."""
    pairs = [("union screen: whole update", "x10_a05__backdoor3__r3wc_shp"),
             ("shared union screen: shared signal", "x11_a05__backdoor3__r3sc")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=False)
    for (title, name), color in zip(pairs, (pr.SLOT[0], pr.SLOT[1]), strict=True):
        m = next(m for m in runs[name] if m["seed"] == 7)
        bad = m["attackers"]
        rounds, zs, gaps, crit, cgaps = [], [], [], [], []
        for d in m["diagnostics"][:8]:
            if set(bad) <= set(d["isolated"]):
                break
            z = _z(d["scores"])
            cz = _z(d["score_stats"]["critical_scores"])
            rounds.append(d["round"])
            zs.append([z.get(c, float("nan")) for c in bad])
            crit.append([cz.get(c, float("nan")) for c in bad])
            gaps.append(d["score_stats"]["gap_ratio"])
            cgaps.append(d["score_stats"]["critical"]["gap_ratio"])
        for k in range(3):
            axes[0].plot(rounds, [z[k] for z in zs], color=color, marker="o", markersize=3, linewidth=1,
                         label=title if k == 0 else None)
        axes[1].plot(rounds, gaps, color=color, marker="o", label=title)
    axes[0].axhline(-2.5, color=pr.INK2, linestyle=":", linewidth=1)
    axes[0].text(1, -2.62, "cut |z| = 2.5", fontsize=7, va="top", color=pr.INK2)
    axes[0].set_title("the three backdoor attackers' z on the whole-update or shared half (seed 7)", fontsize=9.5)
    axes[0].set_xlabel("round")
    axes[1].axhline(8, color=pr.INK2, linestyle=":", linewidth=1)
    axes[1].text(1, 8.3, "gap gate 8", fontsize=7, color=pr.INK2)
    axes[1].set_title("gap ratio of that half (the gate opens at 8)", fontsize=9.5)
    axes[1].set_xlabel("round")
    for ax in axes:
        ax.legend(fontsize=7.5)
    fig.suptitle("Seed 7, α 0.5, 3 backdoor attackers: without the personal heads the attackers stand out less, "
                 "the gate stays shut, and they are caught in round 25 instead of 3–4", fontsize=10, x=0.01,
                 ha="left")
    fig.tight_layout()
    pr.save(fig, "phase8_seed7.png")


def main() -> None:
    pr.style()
    runs = pr.load(Path(sys.argv[1] if len(sys.argv) > 1 else "outputs"))
    res = summarize(runs)
    (ROOT / "docs" / "results" / "phase8.json").write_text(json.dumps(res, indent=1))
    fig_shared(res)
    fig_seed7(runs)
    for lab, arms in res.items():
        print(lab.replace("\n", " "), {a: (len(v), round(statistics.fmean(r["honest"] for r in v), 2) if v else None,
                                          [round(r["asr"], 1) for r in v]) for a, v in arms.items()})


if __name__ == "__main__":
    main()
