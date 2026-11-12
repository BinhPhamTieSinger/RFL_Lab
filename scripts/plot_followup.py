"""Charts for the Phase 5 three-seed results and the 2026-11-12 follow-up questions.

  python scripts/plot_followup.py outputs

Draws into docs/figures/ only the charts whose runs exist."""
from __future__ import annotations

import csv
import importlib.util
import statistics
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location("plot_results", Path(__file__).with_name("plot_results.py"))
pr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pr)
plt = pr.plt
DOCS = Path(__file__).resolve().parents[1] / "docs"

METHODS = [("p2__defended", "Defended FedHEP", "#8c8c8c"), ("p3__zmr", "FedHEP+", pr.SLOT[0]),
           ("p5__median", "Median", "#a9c4e4"), ("p5__trimmed_mean", "Trimmed mean", "#7d9cc0"),
           ("p5__bulyan", "Bulyan", pr.SLOT[1]), ("p5__flame", "FLAME", pr.SLOT[3]),
           ("p5__flame_upd", "FLAME (updates)", "#f5d27a"), ("p5__fltrust", "FLTrust", pr.SLOT[2])]
SETTINGS = [("clean", "No attack"), ("sign_flip", "Sign flip"), ("label_flip", "Label flip"),
            ("noise", "Random noise"), ("backdoor", "Backdoor")]
SEEDS = (42, 123, 7)


def _runs(runs: dict, name: str) -> list[dict]:
    return [m for m in runs.get(name, []) if m["seed"] in SEEDS]


def _honest_flag_rate(m: dict) -> float:
    """Mean share of the 15 - attackers honest clients flagged per round."""
    att = set(m["attackers"])
    return statistics.fmean(len(set(d["flagged"]) - att) / max(1, 15 - len(att)) for d in m["detection"])


def fig_three_seeds(runs: dict) -> None:
    if len(_runs(runs, "p5__fltrust__clean")) < 3:
        return print("skip Phase 5 three seeds")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 4.6), gridspec_kw={"width_ratios": [3.2, 1]})
    w = 0.8 / len(METHODS)
    for j, (pre, label, color) in enumerate(METHODS):
        for i, (s, _) in enumerate(SETTINGS):
            ms = _runs(runs, f"{pre}__{s}")
            if ms:
                mu, sd = pr.mean_sd([m["final"]["pooled"] for m in ms])
                ax.bar(i + (j - 3.5) * w, mu, w * 0.92, yerr=sd, color=color, capsize=1.5,
                       label=label if i == 0 else None, error_kw={"elinewidth": 0.8})
        ms = _runs(runs, f"{pre}__backdoor")
        if ms:
            mu, sd = pr.mean_sd([m["history"][-1]["asr"] for m in ms])
            pr.bar_labels(ax2, ax2.bar(j, mu, 0.75, yerr=sd, color=color, capsize=2), size=7.5)
    ax.set_xticks(range(len(SETTINGS)), [t for _, t in SETTINGS])
    ax.set_ylim(30, 52)
    ax.set_ylabel("Accuracy, honest clients (%)")
    ax.set_title("Accuracy (mean ± sd, 3 seeds, 3 of 15 attackers, α = 0.5)", pad=26)
    ax.legend(ncols=8, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.07), frameon=False)
    ax2.set_xticks(range(len(METHODS)), [lab for _, lab, _ in METHODS], rotation=55, ha="right", fontsize=8)
    ax2.set_ylabel("Backdoor success (%)")
    ax2.set_title("Backdoor success · lower is better")
    fig.tight_layout()
    pr.save(fig, "phase5_three_seeds.png")


def fig_harder(runs: dict) -> None:
    trio = [("p2__defended", "Defended FedHEP", "#8c8c8c"), ("p3__zmr", "FedHEP+", pr.SLOT[0]),
            ("p5__flame", "FLAME", pr.SLOT[3])]
    cases = [("sign_flip", "Sign flip\n3 attackers"), ("sign_flip_5", "Sign flip\n5 attackers"),
             ("backdoor", "Backdoor\n3"), ("backdoor_x3", "Backdoor ×3\n3"), ("backdoor_5", "Backdoor\n5"),
             ("a005_clean", "α = 0.05\nno attack"), ("a005_sign_flip", "α = 0.05\nsign flip 3")]
    if not _runs(runs, "p5__flame__backdoor_5"):
        return print("skip harder settings")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 4.4), gridspec_kw={"width_ratios": [2.4, 1]})
    for j, (pre, label, color) in enumerate(trio):
        for i, (s, _) in enumerate(cases):
            ms = _runs(runs, f"{pre}__{s}")
            if ms:
                mu, sd = pr.mean_sd([m["final"]["pooled"] for m in ms])
                pr.bar_labels(ax, ax.bar(i + (j - 1) * 0.27, mu, 0.25, yerr=sd, color=color, capsize=2,
                                         label=label if i == 0 else None), size=6.5)
        for i, s in enumerate(("backdoor", "backdoor_x3", "backdoor_5")):
            ms = _runs(runs, f"{pre}__{s}")
            if ms:
                mu, sd = pr.mean_sd([m["history"][-1]["asr"] for m in ms])
                pr.bar_labels(ax2, ax2.bar(i + (j - 1) * 0.27, mu, 0.25, yerr=sd, color=color, capsize=2),
                              size=7)
    ax.set_xticks(range(len(cases)), [t for _, t in cases], fontsize=8)
    ax.set_ylim(25, 70)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Harder settings (mean ± sd, 3 seeds)")
    ax.legend(fontsize=8, loc="upper left")
    ax2.set_xticks(range(3), ["3 attackers", "3, boosted ×3", "5 attackers"], fontsize=8)
    ax2.set_ylabel("Backdoor success (%)")
    ax2.set_title("Backdoor success · lower is better")
    fig.tight_layout()
    pr.save(fig, "phase5_harder.png")


def fig_detection(runs: dict) -> None:
    """Precision, recall and false-isolation rate reported separately, per method."""
    attacks = SETTINGS[1:]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for j, (pre, label, color) in enumerate(METHODS):
        for i, (s, _) in enumerate(attacks):
            ms = _runs(runs, f"{pre}__{s}")
            if not ms:
                continue
            x = i + (j - 3.5) * 0.1
            prec = [d["precision"] for m in ms for d in m["detection"] if d.get("precision") is not None]
            rec = [d["recall"] for m in ms for d in m["detection"] if d.get("recall") is not None]
            axes[0].bar(x, statistics.fmean(prec) if prec else 0, 0.09, color=color,
                        label=label if i == 0 else None)
            axes[1].bar(x, statistics.fmean(rec) if rec else 0, 0.09, color=color)
        ms = _runs(runs, f"{pre}__clean") + [m for s, _ in attacks for m in _runs(runs, f"{pre}__{s}")]
        if ms:
            axes[2].bar(j, 100 * statistics.fmean(_honest_flag_rate(m) for m in ms), 0.7, color=color)
    for ax, title in zip(axes[:2], ("Precision: flagged clients that are attackers",
                                    "Recall: attackers flagged"), strict=True):
        ax.set_xticks(range(len(attacks)), [t for _, t in attacks], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=10)
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=8, ncols=8, loc="lower center",
               bbox_to_anchor=(0.5, -0.06), frameon=False)
    axes[2].set_xticks(range(len(METHODS)), [lab for _, lab, _ in METHODS], rotation=55, ha="right", fontsize=8)
    axes[2].set_ylabel("% of honest clients flagged per round")
    axes[2].set_title("False-isolation rate (all 5 settings)", fontsize=10)
    fig.suptitle("Detection, reported separately (mean over rounds and 3 seeds)", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    pr.save(fig, "detection_breakdown.png")


def fig_honest_vs_malicious(runs: dict) -> None:
    """FedHEP+'s screening z per round: attackers vs honest clients."""
    cases = [("sign_flip", "Sign flip"), ("label_flip", "Label flip"), ("noise", "Random noise"),
             ("backdoor", "Backdoor")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    for ax, (s, title) in zip(axes, cases, strict=True):
        for who, color in (("honest", pr.SLOT[0]), ("attackers", pr.SLOT[1])):
            per_round: dict[int, list[float]] = {}
            for m in _runs(runs, f"p3__zmr__{s}"):
                att = {str(c) for c in m["attackers"]}
                for d in m["diagnostics"]:
                    vals = [v for c, v in d["z"].items() if (str(c) in att) == (who == "attackers")]
                    per_round.setdefault(d["round"], []).extend(vals)
            rounds = sorted(r for r, v in per_round.items() if v)
            if not rounds:
                continue
            med = [statistics.median(per_round[r]) for r in rounds]
            lo = [sorted(per_round[r])[int(0.1 * (len(per_round[r]) - 1))] for r in rounds]
            hi = [sorted(per_round[r])[int(0.9 * (len(per_round[r]) - 1))] for r in rounds]
            ax.fill_between(rounds, lo, hi, color=color, alpha=0.2)
            ax.plot(rounds, med, color=color, marker="o" if who == "attackers" else None, markersize=3,
                    label=f"{who} (median, 10–90%)")
        ax.axhline(-2.5, color=pr.INK2, linestyle="--", linewidth=1)
        ax.axhline(-1.5, color=pr.INK2, linestyle=":", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("round")
        ax.set_ylim(-12, 4)
    axes[0].set_ylabel("screening z")
    axes[0].text(30, -2.7, "leave out this round", ha="right", va="top", fontsize=7, color=pr.INK2)
    axes[0].text(30, -1.3, "isolation cut (evidence)", ha="right", va="bottom", fontsize=7, color=pr.INK2)
    axes[0].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("FedHEP+ · how attackers and honest clients score (3 seeds; attackers disappear once isolated)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    pr.save(fig, "honest_vs_malicious_z.png")


def fig_drop(runs: dict) -> None:
    """Accuracy lost to each attack, relative to the same method with no attack."""
    meths = [("p2__fedavg", "FedAvg", "#c9c9c9"), ("p2__fedhep", "FedHEP (no defense)", "#a9c4e4"),
             ("p2__defended", "Defended FedHEP", "#8c8c8c"), ("p3__zmr", "FedHEP+", pr.SLOT[0]),
             ("p5__flame", "FLAME", pr.SLOT[3])]
    attacks = SETTINGS[1:]
    fig, ax = plt.subplots(figsize=(13, 4.2))
    for j, (pre, label, color) in enumerate(meths):
        clean = statistics.fmean(m["final"]["pooled"] for m in _runs(runs, f"{pre}__clean"))
        for i, (s, _) in enumerate(attacks):
            ms = _runs(runs, f"{pre}__{s}")
            if ms:
                d = statistics.fmean(m["final"]["pooled"] for m in ms) - clean
                pr.bar_labels(ax, ax.bar(i + (j - 2) * 0.16, d, 0.15, color=color,
                                         label=label if i == 0 else None), size=7)
    ax.axhline(0, color=pr.INK2, linewidth=1)
    ax.set_xticks(range(len(attacks)), [t for _, t in attacks])
    ax.set_ylabel("Δ accuracy vs no attack (points)")
    ax.set_ylim(-45, 3)
    ax.legend(fontsize=8, ncols=5, loc="lower left")
    ax.set_title("What each attack costs (3 seeds, 3 of 15 attackers, α = 0.5)")
    fig.tight_layout()
    pr.save(fig, "attack_cost.png")


def fig_skew() -> None:
    """Which honest clients get isolated: label skew (offline, from the true labels)."""
    path = DOCS / "followup" / "best_legacy" / "honest_clients.csv"
    if not path.exists():
        return print("skip skew")
    rows = list(csv.DictReader(path.open()))
    groups = [("α = 0.5", lambda r: "a005" not in r["run"]), ("α = 0.05", lambda r: "a005" in r["run"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, (title, keep) in zip(axes, groups, strict=True):
        sel = [r for r in rows if keep(r)]
        for k, (flag, color) in enumerate((("False", pr.SLOT[0]), ("True", pr.SLOT[1]))):
            v = [float(r["r_skew"]) for r in sel if r["isolated"] == flag]
            if v:
                jitter = [k + 0.25 * ((i * 0.618) % 1 - 0.5) for i in range(len(v))]
                ax.scatter(jitter, v, s=10, color=color, alpha=0.6)
                ax.plot([k - 0.3, k + 0.3], [statistics.median(v)] * 2, color=pr.INK, linewidth=2)
                ax.text(k + 0.32, statistics.median(v), f"median {statistics.median(v):.2f}\nn = {len(v)}",
                        fontsize=7.5, va="center")
        ax.set_xticks([0, 1], ["kept", "isolated"])
        ax.set_xlim(-0.5, 1.9)
        ax.set_ylabel("R_skew (lower = more skewed)")
        ax.set_title(f"Honest clients, FedHEP+, {title}")
    fig.tight_layout()
    pr.save(fig, "skew_vs_isolation.png")


def main() -> None:
    runs = pr.load(Path(sys.argv[1]))
    pr.style()
    fig_three_seeds(runs)
    fig_harder(runs)
    fig_detection(runs)
    fig_honest_vs_malicious(runs)
    fig_drop(runs)
    fig_skew()
    fig_why_z_grows(runs)



def _lower_median(v: list[float]) -> float:
    v = sorted(v)
    return v[(len(v) - 1) // 2]  # torch.median's convention, as screening uses


def fig_why_z_grows(runs: dict) -> None:
    """Why honest clients' z gets more extreme mid-training, read from FedHEP+'s logs (no raw scores were logged):
    MAD(z) < 0.674 means the relative floor 0.1|median cosine| set the scale that round; = 0.674 means the raw MAD
    did. SCCF's logged weights are softmax(cos / T), so T ln w recovers cosines up to a constant (T ~ 0.5): their
    spread, and whether SCCF fell back to equal weights (no consensus)."""
    import math
    names = [f"p3__zmr__{s}" for s, _ in SETTINGS]
    ms = [m for n in names for m in _runs(runs, n)]
    if not ms:
        return print("skip why z grows")
    rounds = range(1, 31)
    floor_binds, spread, fallback, new_iso = ({r: [] for r in rounds} for _ in range(4))
    for m in ms:
        att, seen = set(m["attackers"]), set()
        for d in m["diagnostics"]:
            r, ref = d["round"], list(d["z"].values())
            if len(ref) >= 5:
                med = _lower_median(ref)
                floor_binds[r].append(_lower_median([abs(v - med) for v in ref]) < 0.6745 - 1e-3)
            w = [v for v in d["global_trust"].values() if v > 0]
            if len(w) >= 3:
                uniform = max(w) - min(w) < 1e-6
                fallback[r].append(uniform)
                if not uniform:
                    c = [0.5 * math.log(v) for v in w]
                    cm = statistics.median(c)
                    spread[r].append(1.4826 * statistics.median([abs(x - cm) for x in c]))
            iso = set(d["isolated"]) - att
            new_iso[r].append(len(iso - seen))
            seen |= iso
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.9))
    axes[0].plot(rounds, [100 * statistics.fmean(floor_binds[r]) for r in rounds], color=pr.SLOT[0], marker="o",
                 markersize=3, label="floor 0.1·|median cos| sets the scale")
    axes[0].plot(rounds, [100 * statistics.fmean(fallback[r]) if fallback[r] else 0 for r in rounds],
                 color=pr.SLOT[1], marker="s", markersize=3, label="SCCF: no consensus (equal weights)")
    axes[0].set_ylabel("% of the 15 FedHEP+ runs")
    axes[0].set_title("What sets z's denominator")
    axes[0].legend(fontsize=7.5)
    axes[1].plot(rounds, [statistics.median(spread[r]) if spread[r] else float("nan") for r in rounds],
                 color=pr.SLOT[2], marker="o", markersize=3)
    axes[1].set_ylabel("spread of cosines (1.4826·MAD, proxy)")
    axes[1].set_title("Honest disagreement grows")
    axes[2].bar(list(rounds), [sum(new_iso[r]) for r in rounds], color=pr.SLOT[3])
    axes[2].set_ylabel("honest clients newly isolated (15 runs)")
    axes[2].set_title("When false isolations happen")
    for ax in axes:
        ax.set_xlabel("round")
    fig.suptitle("Why honest z-scores get extreme mid-training (FedHEP+, 5 settings × 3 seeds)", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    pr.save(fig, "why_z_grows.png")


if __name__ == "__main__":
    main()
