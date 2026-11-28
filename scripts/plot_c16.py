"""Acer c16: readmission (literature review Part 2, experiment 4; P51-P56).

  python scripts/plot_c16.py outputs

Runs: acer_c16_readmission.yaml (x16_a05__*), 15 clients, alpha 0.5, 3 attackers, FedHEP-Guard's screen with
permanent isolation (r3wc), Phase 3b readmission (r3wc_ra), a 5-round sentence (r3wc_s5) and the same with a
1-round probation patience (r3wc_s5p1). Writes docs/figures/acer_readmission.png and docs/results/acer_c16.json."""
from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("plot_results", Path(__file__).with_name("plot_results.py"))
pr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pr)
plt = pr.plt
SEEDS = (42, 123, 7)
MODELS = [("r3wc", "permanent isolation (union screen)"), ("r3wc_ra", "readmission when evidence recovers (3b)"),
          ("r3wc_s5", "5-round sentence"), ("r3wc_s5p1", "5-round sentence, 1-round probation")]
CONDITIONS = [("clean", "no attack"), ("backdoor3", "backdoor"), ("sleeper3", "sleeper backdoor\n(from round 10)"),
              ("onoff3", "on-off backdoor\n(4 on, 6 off)"), ("onoff21", "on-off backdoor\n(2 on, 1 off)")]


def run(m: dict) -> dict:
    bad = set(m["attackers"])
    first: dict[int, int] = {}
    prev: set[int] = set()
    for d in m["diagnostics"]:
        cur = set(d["isolated"])
        for c in cur - prev:
            first.setdefault(c, d["round"])
        prev = cur
    return {"seed": m["seed"], "acc": m["final"]["pooled"], "bottom10": m["final"]["bottom10"],
            "asr": m["history"][-1].get("asr") or 0.0, "honest_end": len(prev - bad), "honest_ever": len(set(first) - bad),
            "attackers_end": len(prev & bad), "attacker_first": sorted(first[c] for c in first if c in bad)}


def main() -> None:
    pr.style()
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs")
    res: dict = {}
    for cond, _ in CONDITIONS:
        for mdl, _ in MODELS:
            rows = []
            for s in SEEDS:
                p = out / f"x16_a05__{cond}__{mdl}_s{s}" / "metrics.json"
                if p.exists():
                    m = json.loads(p.read_text())
                    if m.get("final"):
                        rows.append(run(m))
            res[f"{cond}/{mdl}"] = rows
    (ROOT / "docs" / "results" / "acer_c16.json").write_text(json.dumps(res, indent=1))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1, 1, 2.6]})
    w = 0.8 / len(MODELS)
    colors = [pr.SLOT[0], pr.SLOT[3], pr.SLOT[1], pr.SLOT[2]]
    for ax, field, title in ((axes[0], "honest_end", "honest clients isolated at the end, no attack"),
                             (axes[1], "acc", "accuracy (%), no attack")):
        for j, (mdl, name) in enumerate(MODELS):
            rows = res[f"clean/{mdl}"]
            if not rows:
                continue
            v = statistics.fmean(r[field] for r in rows)
            ax.bar(j, v, 0.7, color=colors[j], label=name)
            ax.text(j, v, f"{v:.1f}\nn={len(rows)}", ha="center", va="bottom", fontsize=7)
            for r in rows:
                ax.scatter(j, r[field], color=pr.INK, s=10, zorder=3)
        ax.set_xticks([])
        ax.set_title(title, fontsize=9.5)
    axes[1].set_ylim(40, 50)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[2].legend(handles, labels, fontsize=7.5, loc="upper right")
    ax = axes[2]
    attacked = CONDITIONS[1:]
    for j, (mdl, _name) in enumerate(MODELS):
        for i, (cond, _) in enumerate(attacked):
            rows = res[f"{cond}/{mdl}"]
            x = i + (j - 1.5) * w
            if not rows:
                ax.text(x, 2, "not run\nyet", ha="center", fontsize=6, color=pr.INK2)
                continue
            ax.bar(x, statistics.fmean(r["asr"] for r in rows), w * 0.95, color=colors[j], alpha=0.85)
            for r in rows:
                ax.scatter(x, r["asr"], color=pr.INK, s=10, zorder=3)
    ax.set_xticks(range(len(attacked)), [t for _, t in attacked], fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_title("backdoor success (%), dots = seeds", fontsize=9.5)
    fig.suptitle("Acer c16, readmission at 15 clients: a 5-round sentence brings the honest clients back, but only a "
                 "1-round probation keeps a returning backdoor attacker out; a sleeper backdoor beats readmission",
                 fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    pr.save(fig, "acer_readmission.png")
    for k, rows in res.items():
        if rows:
            print(k, len(rows), [(r["seed"], round(r["asr"], 1), r["honest_end"], r["attackers_end"]) for r in rows])


if __name__ == "__main__":
    main()
