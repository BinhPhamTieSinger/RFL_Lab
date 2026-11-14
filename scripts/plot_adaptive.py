"""Charts for the 2026-11-13 follow-up runs: adaptive attacks and screening diagnostics.

Usage: python scripts/plot_adaptive.py outputs outputs_followup_full
"""
from __future__ import annotations

import importlib.util
import statistics
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location("plot_results", Path(__file__).with_name("plot_results.py"))
assert _spec and _spec.loader
pr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pr)
plt = pr.plt

METHODS = [("p2__fedhep", "FedHEP (no defense)", "#b0b0b0"), ("p2__defended", "Defended FedHEP", "#6e6e6e"),
           ("p3__zmr", "FedHEP+", pr.SLOT[0]), ("p5__bulyan", "Bulyan", pr.SLOT[1]),
           ("p5__flame", "FLAME", pr.SLOT[3])]
ATTACKS = [("clean", "No attack"), ("alie3", "ALIE (3 of 15)"), ("minmax3", "Min-Max (3 of 15)")]
VARIANTS = [("baseline", "FedHEP+"), ("floor001", "floor 0.01"), ("floor003", "floor 0.03"),
            ("patience3", "patience 3")]
SEEDS = (42, 123, 7)


def _runs(runs: dict, name: str) -> list[dict]:
    return [m for m in runs.get(name, []) if m["seed"] in SEEDS]


def _false_isolations(m: dict) -> int:
    return len(set(m["diagnostics"][-1].get("isolated", [])) - set(m["attackers"]))


def fig_adaptive(runs: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, (attack, _) in enumerate(ATTACKS):
        for j, (prefix, _label, color) in enumerate(METHODS):
            ms = _runs(runs, f"{prefix}__{attack}")
            if ms:
                mu, sd = pr.mean_sd([m["final"]["pooled"] for m in ms])
                pr.bar_labels(ax, ax.bar(i + (j - 2) * 0.16, mu, 0.15, yerr=sd, color=color, capsize=2,
                                         label=_label if i == 0 else None), size=7.5, errors=[sd])
    ax.set_xticks(range(len(ATTACKS)), [label for _, label in ATTACKS])
    ax.set_ylim(30, 52)
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Adaptive attacks: accuracy (mean ± sd, 3 seeds, α = 0.5); no method isolated any attacker")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(*ax.get_legend_handles_labels(), ncols=5, fontsize=8.5, loc="lower center", frameon=False)
    pr.save(fig, "followup_adaptive.png")


def fig_diagnostics(diag: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.1))
    settings = [("clean", "No attack"), ("sign3", "Sign flip"), ("backdoor3", "Backdoor")]
    for j, (variant, label) in enumerate(VARIANTS):
        counts = [sum(_false_isolations(m) for m in _runs(diag, f"followup__{variant}__{s}"))
                  for s, _ in settings]
        pr.bar_labels(axes[0], axes[0].bar([i + (j - 1.5) * 0.2 for i in range(3)], counts, 0.19,
                                           color=pr.SLOT[j], label=label), fmt="{:.0f}", size=7.5)
        asr = [m["history"][-1]["asr"] for m in _runs(diag, f"followup__{variant}__backdoor3")]
        mu, sd = pr.mean_sd(asr)
        pr.bar_labels(axes[2], axes[2].bar(j, mu, 0.6, yerr=sd, color=pr.SLOT[j], capsize=2), size=7.5,
                      errors=[sd])
    axes[0].set_xticks(range(3), [label for _, label in settings])
    axes[0].set_ylabel("honest clients isolated (sum of 3 seeds)")
    axes[0].set_title("False isolations")
    axes[0].legend(fontsize=8)
    base = [m for s, _ in settings for m in _runs(diag, f"followup__baseline__{s}")]
    rounds = range(1, 31)
    stats = [[m["diagnostics"][r - 1]["score_stats"] for m in base] for r in rounds]
    axes[1].plot(rounds, [1.4826 * statistics.median(s["mad"] for s in row) for row in stats], color=pr.SLOT[0],
                 marker="o", markersize=3, label="1.4826 · MAD")
    axes[1].plot(rounds, [statistics.median(s["relative_floor"] for s in row) for row in stats], color=pr.SLOT[1],
                 marker="s", markersize=3, label="relative floor 0.1·|median|")
    axes[1].set_xlabel("round")
    axes[1].set_ylabel("median over 9 FedHEP+ runs")
    axes[1].set_title("Measured MAD grows; it does not shrink")
    axes[1].legend(fontsize=8)
    axes[2].set_xticks(range(len(VARIANTS)), [label for _, label in VARIANTS])
    axes[2].set_ylabel("backdoor success (%)")
    axes[2].set_title("Backdoor success (mean ± sd, 3 seeds)")
    fig.tight_layout()
    pr.save(fig, "followup_diagnostics.png")


def main() -> None:
    pr.style()
    fig_adaptive(pr.load(Path(sys.argv[1])))
    fig_diagnostics(pr.load(Path(sys.argv[2])))


if __name__ == "__main__":
    main()
