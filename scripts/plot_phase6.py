"""Charts for the first Phase 6 results (2026-11-15): two-sided screening, layer L0, the backdoor-stop test and
30 clients.

Usage: python scripts/plot_phase6.py outputs
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location("plot_results", Path(__file__).with_name("plot_results.py"))
assert _spec and _spec.loader
pr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pr)
plt = pr.plt
SEEDS = (42, 123, 7)


def _runs(runs: dict, name: str) -> list[dict]:
    return [m for m in runs.get(name, []) if m["seed"] in SEEDS]


def _false(m: dict) -> int:
    return len(set(m["diagnostics"][-1].get("isolated", [])) - set(m["attackers"])) if m.get("diagnostics") else 0


def _grouped(ax, runs, methods, attacks, width):
    for j, (prefix, label, color) in enumerate(methods):
        for i, (a, _) in enumerate(attacks):
            ms = _runs(runs, f"{prefix}__{a}")
            if not ms:
                continue
            mu, sd = pr.mean_sd([m["final"]["pooled"] for m in ms])
            pr.bar_labels(ax, ax.bar(i + (j - (len(methods) - 1) / 2) * width, mu, width - 0.02, yerr=sd,
                                     color=color, capsize=2, label=label if i == 0 else None), size=7, errors=[sd])
    ax.set_xticks(range(len(attacks)), [label for _, label in attacks])
    ax.set_ylabel("accuracy (%)")


def fig_d1(runs: dict) -> None:
    attacks = [("clean", "No attack"), ("sign_flip", "Sign flip"), ("backdoor", "Backdoor"), ("alie3", "ALIE"),
               ("minmax3", "Min-Max")]
    methods = [("p2__defended", "Defended FedHEP", "#6e6e6e"), ("p3__zmr", "FedHEP+", pr.SLOT[0]),
               ("ts__two_p3", "FedHEP+ two-sided, patience 3", pr.SLOT[2]),
               ("p6__l0", "L0 (ARC, NNM, trimmed mean)", pr.SLOT[1]),
               ("p6__l0m", "L0 + momentum 0.5", pr.SLOT[3])]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 4.4), gridspec_kw={"width_ratios": [2.6, 1]})
    _grouped(ax, runs, methods, attacks, 0.16)
    ax.set_ylim(36, 52)
    ax.set_title("D1: α = 0.5, 15 clients, 3 attackers (mean ± sd, 3 seeds)")
    for j, (prefix, _label, color) in enumerate(methods[1:3]):
        att = [sum(len(set(m["diagnostics"][-1].get("isolated", [])) & set(m["attackers"]))
                   for m in _runs(runs, f"{prefix}__{a}")) for a, _ in attacks]
        pr.bar_labels(ax2, ax2.bar([i + (j - 0.5) * 0.38 for i in range(5)], att, 0.36, color=color), fmt="{:.0f}",
                      size=7.5)
    ax2.set_xticks(range(5), [label for _, label in attacks], fontsize=7.5)
    ax2.set_ylabel("attackers isolated (of 9 over 3 seeds)")
    ax2.set_title("One-sided vs two-sided screening")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.legend(*ax.get_legend_handles_labels(), ncols=5, fontsize=8.5, loc="lower center", frameon=False)
    pr.save(fig, "phase6_d1.png")


def fig_d2(runs: dict) -> None:
    attacks = [("clean", "No attack"), ("sign4", "Sign flip"), ("backdoor4", "Backdoor"), ("minmax4", "Min-Max")]
    methods = [("a02__fedhep", "FedHEP (no defense)", "#b0b0b0"), ("a02__defended", "Defended FedHEP", "#6e6e6e"),
               ("a02__plus", "FedHEP+", pr.SLOT[0]),
               ("a02__plus_p3_two", "FedHEP+ two-sided, patience 3", pr.SLOT[2]),
               ("a02__l0", "L0", pr.SLOT[1])]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 4.4), gridspec_kw={"width_ratios": [2.4, 1]})
    _grouped(ax, runs, methods, attacks, 0.16)
    ax.set_ylim(40, 58)
    ax.set_title("D2: α = 0.2, 20 clients, 4 attackers (mean ± sd, 3 seeds)")
    variants = [methods[2], ("a02__plus_p3", "FedHEP+ patience 3", "#9a9a9a"), methods[3]]
    for j, (prefix, label, color) in enumerate(variants):
        hon = [sum(_false(m) for m in _runs(runs, f"{prefix}__{a}")) for a, _ in attacks]
        pr.bar_labels(ax2, ax2.bar([i + (j - 1) * 0.27 for i in range(4)], hon, 0.25, color=color, label=label),
                      fmt="{:.0f}", size=7.5)
    ax2.set_xticks(range(4), [label for _, label in attacks], fontsize=7.5)
    ax2.set_ylabel("honest clients isolated (sum of 3 seeds)")
    ax2.set_title("Screening variants: false isolations (L0 isolates nobody)")
    ax2.legend(fontsize=7.5)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.legend(*ax.get_legend_handles_labels(), ncols=5, fontsize=8.5, loc="lower center", frameon=False)
    pr.save(fig, "phase6_d2.png")


def fig_backdoor_stop(runs: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9), sharey=True)
    series = [("n20__fedhep__backdoor5", "attackers poison all 30 rounds", "#4a4a4a"),
              ("n20__fedhep__backdoor5_stop15", "attackers stop after round 15", pr.SLOT[1]),
              ("n20__fedhep__backdoor5_stop10", "attackers stop after round 10", pr.SLOT[0])]
    for ax, seed in zip(axes, SEEDS, strict=True):
        for name, label, color in series:
            m = next(x for x in runs[name] if x["seed"] == seed)
            rows = [h for h in m["history"] if h.get("asr") is not None]
            ax.plot([h["round"] for h in rows], [h["asr"] for h in rows], marker="o", markersize=3.5, color=color,
                    label=label)
        for r, c in ((10, pr.SLOT[0]), (15, pr.SLOT[1])):
            ax.axvline(r, color=c, linestyle="--", linewidth=1)
        ax.set_title(f"seed {seed}")
        ax.set_xlabel("round")
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("backdoor success on honest clients (%)")
    fig.suptitle("Undefended FedHEP, 20 clients, 5 backdoor attackers: success keeps rising after poisoning stops",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(*axes[0].get_legend_handles_labels(), ncols=3, fontsize=8.5, loc="lower center", frameon=False)
    pr.save(fig, "followup_backdoor_stop.png")


def fig_n30(runs: dict) -> None:
    attacks = [("clean", "No attack"), ("sign6", "6 sign flippers (20%)"), ("sign9", "9 (30%)"),
               ("sign12", "12 (40%)")]
    methods = [("n30__fedavg", "FedAvg", "#b0b0b0"), ("n30__fedhep", "FedHEP (no defense)", "#8a8a8a"),
               ("n30__defended", "Defended FedHEP", "#4a4a4a"), ("n30__plus", "FedHEP+", pr.SLOT[0])]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    _grouped(ax, runs, methods, attacks, 0.2)
    ax.set_ylim(0, 50)
    ax.set_title("30 clients, α = 0.5, sign flip (mean ± sd over the seeds finished; FedHEP+ at 9 and 12 pending)")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.legend(*ax.get_legend_handles_labels(), ncols=4, fontsize=8.5, loc="lower center", frameon=False)
    pr.save(fig, "followup_n30.png")


def main() -> None:
    pr.style()
    runs = pr.load(Path(sys.argv[1]))
    fig_d1(runs)
    fig_d2(runs)
    fig_backdoor_stop(runs)
    fig_n30(runs)


if __name__ == "__main__":
    main()
