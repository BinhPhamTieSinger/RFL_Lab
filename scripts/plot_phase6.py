"""Charts for the first Phase 6 results (2026-11-15): two-sided screening, layer L0, the backdoor-stop test and
30 clients.

Usage: python scripts/plot_phase6.py outputs
"""
from __future__ import annotations

import importlib.util
import json
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
               ("a02__l0", "L0", pr.SLOT[1]),
               ("a02__r2", "R2: L0 + two-sided", pr.SLOT[2]),
               ("a02__r3g8", "R3: gate 8", pr.SLOT[3]),
               ("a02__r3g12", "R3: gate 12", "#7a3fb8")]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 4.4), gridspec_kw={"width_ratios": [2.4, 1]})
    _grouped(ax, runs, methods, attacks, 0.12)
    ax.set_ylim(40, 58)
    ax.set_title("D2: α = 0.2, 20 clients, 4 attackers (mean ± sd, 3 seeds)")
    variants = [methods[2], methods[4], methods[5], methods[6]]
    for j, (prefix, label, color) in enumerate(variants):
        hon = [sum(_false(m) for m in _runs(runs, f"{prefix}__{a}")) for a, _ in attacks]
        pr.bar_labels(ax2, ax2.bar([i + (j - 1.5) * 0.21 for i in range(4)], hon, 0.19, color=color, label=label),
                      fmt="{:.0f}", size=7.5)
    ax2.set_xticks(range(4), [label for _, label in attacks], fontsize=7.5)
    ax2.set_ylabel("honest clients isolated (sum of 3 seeds)")
    ax2.set_title("False isolations by rung (L0 isolates nobody)")
    ax2.legend(fontsize=7.5)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.legend(*ax.get_legend_handles_labels(), ncols=7, fontsize=8.5, loc="lower center", frameon=False)
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


def fig_backdoor_heads(runs: dict) -> None:
    """Backdoor success per prediction head, by rung (mean over seeds): where the backdoor lives."""
    rungs = [("p6__r2", "R2 (no gate)"), ("p6__r3g8", "R3, gate 8"), ("p6__r3g12", "R3, gate 12")]
    heads = [("root", "Root head"), ("parent", "Parent head"), ("local", "Local head"), ("asr", "Ensemble")]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    width = 0.2
    for j, (head, label) in enumerate(heads):
        ys, es = [], []
        for prefix, _ in rungs:
            rows = [r for name, rs in runs.items() if name.startswith(prefix + "__backdoor") for r in rs]
            last = [r["history"][-1] for r in rows]
            vals = [x["asr"] if head == "asr" else x["asr_heads"][head] for x in last]
            m, s = pr.mean_sd(vals)
            ys.append(m)
            es.append(s)
        xs = [i + (j - 1.5) * width for i in range(len(rungs))]
        pr.bar_labels(ax, ax.bar(xs, ys, width * 0.92, yerr=es, color=pr.SLOT[j], label=label, capsize=2),
                      fmt="{:.0f}", errors=es)
    ax.set_xticks(range(len(rungs)), [label for _, label in rungs])
    ax.set_ylabel("Backdoor success at round 30 (%)")
    ax.set_ylim(0, 100)
    ax.legend(ncol=4, frameon=False, loc="upper left", fontsize=8)
    ax.set_title("D1 backdoor, 3 attackers: the root head carries the backdoor, the local head resists it",
                 fontsize=10)
    pr.save(fig, "phase6_backdoor_heads")


def fig_durability(runs: dict) -> None:
    """100 rounds, backdoor stopped after round 50 (acer_c6_durability): backdoor success over time per method,
    and with the depth-exit guard at prediction time (dashed). Whatever x6 runs exist are drawn."""
    methods = [("x6__fedhep", "FedHEP (no defense)", "#6b6b6b"), ("x6__l0", "L0 alone", pr.SLOT[1]),
               ("x6__plus", "FedHEP+ (isolates early)", pr.SLOT[0]), ("x6__r3g8", "R3, gate 8", pr.SLOT[2]),
               ("x6__r3_tr", "R3 + rewind", pr.SLOT[3]), ("x6__r3w", "R3 whole-update", "#b48ad8"),
               ("x6__r3w_tr", "R3 whole-update + rewind", "#7a3fb8")]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for prefix, label, color in methods:
        done = [m for m in runs.get(prefix, []) if m.get("final")]
        if not done:
            continue
        rounds = [r["round"] for r in done[0]["history"]]
        mean = [sum(m["history"][i]["asr"] for m in done) / len(done) for i in range(len(rounds))]
        ax.plot(rounds, mean, color=color, label=f"{label} (n={len(done)})")
        if all(m["history"][-1].get("guarded") for m in done) and prefix != "x6__plus":
            guard = [sum(m["history"][i]["guarded"]["asr"] for m in done) / len(done) for i in range(len(rounds))]
            ax.plot(rounds, guard, color=color, linestyle="--", linewidth=1.4, label=f"{label} + exit guard")
    ax.axvline(50, color=pr.INK2, linewidth=1, linestyle=":")
    ax.text(51, 92, "attackers stop", fontsize=8, color=pr.INK2)
    ax.set_xlabel("round")
    ax.set_ylabel("backdoor success (%)")
    ax.set_ylim(0, 100)
    ax.set_title("A backdoor outlives its attackers unless they are caught early (D1, 3 attackers, Acer)")
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    pr.save(fig, "phase6_durability.png")


def fig_rewind(runs: dict) -> None:
    """Per seed, 100 rounds (acer_c6_durability): does rewinding the trunk and root head when an attacker is
    isolated remove a backdoor that got in before the catch? Solid: no rewind; dashed: rewind."""
    panels = [("root-head signal (R3, gate 8)", "x6__r3g8", "x6__r3_tr"),
              ("whole-update signal (R3)", "x6__r3w", "x6__r3w_tr")]
    if not all(len(_runs(runs, n)) == 3 for _, a, b in panels for n in (a, b)):
        return
    colors = dict(zip(SEEDS, (pr.SLOT[0], pr.SLOT[1], pr.SLOT[2]), strict=True))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), sharey=True)
    for ax, (title, plain, rewind) in zip(axes, panels, strict=True):
        for name, style in ((plain, "-"), (rewind, "--")):
            for m in _runs(runs, name):
                rows = [h for h in m["history"] if h.get("asr") is not None]
                ax.plot([h["round"] for h in rows], [h["asr"] for h in rows], linestyle=style,
                        color=colors[m["seed"]], linewidth=1.8,
                        label=f"seed {m['seed']}, {'rewind' if name == rewind else 'no rewind'}")
        ax.axvline(50, color=pr.INK2, linewidth=1, linestyle=":")
        ax.text(51, 92, "attackers stop", fontsize=8, color=pr.INK2)
        ax.set_title(title)
        ax.set_xlabel("round")
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("backdoor success on honest clients (%)")
    fig.suptitle("Rewind on isolation, 100 rounds (D1, 3 backdoor attackers until round 50, Acer)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(*axes[1].get_legend_handles_labels(), ncols=3, fontsize=8.5, loc="lower center", frameon=False)
    pr.save(fig, "phase6_rewind.png")


def fig_stealth(runs: dict) -> None:
    """mac_c7_stealth: backdoor success per method for the plain backdoor and the stealth backdoor (blended toward
    the honest mean inside Min-Max's bound), 3 and 5 attackers. Bars are means; dots are the three seeds."""
    conds = [("backdoor3", "plain backdoor, 3 attackers", pr.SLOT[0]),
             ("stealth3", "stealth backdoor, 3 attackers", pr.SLOT[1]),
             ("stealth5", "stealth backdoor, 5 attackers", pr.SLOT[2])]
    methods = [("fedhep", "FedHEP"), ("defended", "Defended\nFedHEP"), ("l0", "L0"), ("plus", "FedHEP+"),
               ("r3w", "R3\nwhole update"), ("r3wc", "R3\nunion screen"), ("flame", "FLAME")]
    if not all(len(_runs(runs, f"m7__{c}__{m}")) == 3 for c, _, _ in conds for m, _ in methods):
        return
    fig, ax = plt.subplots(figsize=(12, 4.2))
    width = 0.26
    for j, (cond, label, color) in enumerate(conds):
        for i, (m, _) in enumerate(methods):
            vals = [r["history"][-1]["asr"] for r in _runs(runs, f"m7__{cond}__{m}")]
            x = i + (j - 1) * width
            ax.bar(x, sum(vals) / 3, width - 0.03, color=color, alpha=0.85, label=label if i == 0 else None)
            ax.scatter([x] * 3, vals, s=12, color=pr.INK, zorder=3)
    ax.set_xticks(range(len(methods)), [label for _, label in methods], fontsize=8.5)
    ax.set_ylabel("backdoor success at round 30 (%)")
    ax.set_ylim(0, 100)
    ax.set_title("A backdoor that hides inside the honest spread (D1, Mac): only the union screen and FLAME stay under "
                 "10% on every seed", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(*ax.get_legend_handles_labels(), ncols=3, fontsize=8.5, loc="lower center", frameon=False)
    pr.save(fig, "phase6_stealth.png")


def fig_fmis(runs: dict) -> None:
    """acer_c8_fmis: L0 and R3 (union screen) when the defender assumes f = 1, 3 or 6 while 3 attack (or none)."""
    fs = (1, 3, 6)
    methods = [("l0", "L0", pr.SLOT[1]), ("r3wc", "R3, union screen", pr.SLOT[0])]
    panels = [("clean", "No attack"), ("sign3", "Sign flip"), ("alie3", "ALIE"), ("backdoor3", "Backdoor")]
    if not all(len(_runs(runs, f"x8_f{f}__{a}__{m}")) == 3 for f in fs for a, _ in panels for m, _, _ in methods):
        return
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, (attack, title) in zip(axes, panels, strict=True):
        for m, label, color in methods:
            if attack == "backdoor3":
                stats = [pr.mean_sd([r["history"][-1]["asr"] for r in _runs(runs, f"x8_f{f}__{attack}__{m}")])
                         for f in fs]
            else:
                stats = [pr.mean_sd([r["final"]["pooled"] for r in _runs(runs, f"x8_f{f}__{attack}__{m}")])
                         for f in fs]
            ax.errorbar(fs, [mu for mu, _ in stats], yerr=[sd for _, sd in stats], color=color, marker="o",
                        capsize=3, label=label)
        ax.axvline(3, color=pr.INK2, linewidth=1, linestyle=":")
        ax.set_xticks(fs, ["1", "3\n(true)", "6"])
        ax.set_xlabel("attackers the defender assumes (f)")
        ax.set_title(title)
        ax.set_ylabel("backdoor success (%)" if attack == "backdoor3" else "accuracy (%)")
        if attack == "backdoor3":
            ax.set_ylim(0, 100)
        else:
            ax.set_ylim(33, 50)
    fig.suptitle("Mis-specified f (D1, 3 attackers, Acer, mean ± sd over 3 seeds): the screen makes R3 indifferent to "
                 "f; L0 depends on it", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(*axes[0].get_legend_handles_labels(), ncols=2, fontsize=8.5, loc="lower center", frameon=False)
    pr.save(fig, "phase6_fmis.png")


def fig_signals(runs: dict) -> None:
    """GPU batch gpu_d1_signals (CUDA, D1, 3 seeds): which screening signal catches the backdoor, and what each
    costs in honest clients isolated with no attack."""
    rows = [("g1__fedhep", "FedHEP"), ("g1__l0", "L0"), ("g1__plus", "FedHEP+"), ("g1__r3", "R3 root head"),
            ("g1__r3w", "R3 whole update"), ("g1__r3c", "R3 critical layers"), ("g1__r3s", "R3 sign align"),
            ("g1__r3p", "R3 probe logits"), ("g1__r3h", "R3 head gap")]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
    for i, (prefix, _label) in enumerate(rows):
        bd = _runs(runs, f"{prefix}__backdoor3")
        ys = [m["history"][-1]["asr"] for m in bd]
        ax.scatter([i] * len(ys), ys, color=pr.SLOT[0], s=28, zorder=3)
        if ys:
            ax.hlines(sum(ys) / len(ys), i - 0.3, i + 0.3, color=pr.INK, linewidth=2)
        guard = [m["history"][-1]["guarded"]["asr"] for m in bd if m["history"][-1].get("guarded")]
        ax.scatter([i + 0.18] * len(guard), guard, color=pr.SLOT[1], s=16, marker="D", zorder=3,
                   label="with exit guard" if i == 0 else None)
        clean = _runs(runs, f"{prefix}__clean")
        hon = sum(_false(m) for m in clean)
        acc = sum(m["final"]["pooled"] for m in clean) / max(1, len(clean))
        ax2.barh(i, hon, color=pr.SLOT[2] if hon <= 1 else pr.SLOT[3])
        ax2.text(hon + 0.2, i, f"{hon}   ({acc:.1f}% acc.)", va="center", fontsize=8, color=pr.INK2)
    ax.set_xticks(range(len(rows)), [r[1] for r in rows], rotation=30, ha="right")
    ax.set_ylabel("backdoor success at round 30 (%)")
    ax.set_ylim(-3, 100)
    ax.scatter([], [], color=pr.SLOT[0], s=28, label="each seed")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Backdoor (3 attackers): no single signal catches it on every seed")
    ax2.set_yticks(range(len(rows)), [r[1] for r in rows])
    ax2.invert_yaxis()
    ax2.set_xlabel("honest clients isolated with no attack (sum over 3 seeds)")
    ax2.set_xlim(0, 16)
    ax2.set_title("Cost with no attack")
    pr.save(fig, "phase6_signals.png")


def fig_devices(pairs_path: Path) -> None:
    """Same configuration and seed on the Mac (MPS) and the GPU server (CUDA): final accuracy of each pair."""
    if not pairs_path.exists():
        return
    pairs = [p for p in json.loads(pairs_path.read_text()) if "mps" in p and "cuda" in p]
    xs = [p["mps"][1] for p in pairs]
    ys = [p["cuda"][1] for p in pairs]
    diffs = [y - x for x, y in zip(xs, ys, strict=True)]
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.scatter(xs, ys, s=18, color=pr.SLOT[0], alpha=0.8)
    lo, hi = min(xs + ys) - 0.5, max(xs + ys) + 0.5
    ax.plot([lo, hi], [lo, hi], color=pr.INK2, linewidth=1, linestyle=":")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Mac (MPS) final accuracy (%)")
    ax.set_ylabel("GPU server (CUDA) final accuracy (%)")
    mean_abs = sum(map(abs, diffs)) / len(diffs)
    ax.set_title(f"{len(pairs)} identical runs on two devices: mean |difference| {mean_abs:.2f} points",
                 fontsize=10)
    pr.save(fig, "phase6_devices.png")


def main() -> None:
    pr.style()
    runs = pr.load(Path(sys.argv[1]))
    fig_d1(runs)
    fig_d2(runs)
    fig_backdoor_stop(runs)
    fig_n30(runs)
    fig_backdoor_heads(runs)
    fig_durability(runs)
    fig_rewind(runs)
    fig_stealth(runs)
    fig_fmis(runs)
    fig_signals(runs)
    if len(sys.argv) > 2:
        fig_devices(Path(sys.argv[2]))


if __name__ == "__main__":
    main()
