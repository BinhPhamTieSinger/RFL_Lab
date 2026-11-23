"""Charts and tables for the canonical benchmark (report chapter 7), drawn only from the benchmark runs
(bench_fedhep_family.yaml and bench_fedavg_family.yaml, GPU server): every model, the same attacks, one machine.

  python scripts/plot_benchmark.py outputs           # figures in docs/report/figures/, tables in chapter 7

Run names are bm_<setting>__<attack>__<model>_s<seed>. A cell with fewer than three seeds is drawn with the seeds
it has and marked in the tables with its seed count. The tables in docs/report/09_benchmark.md are rewritten
between the markers <!-- BENCH:<setting> START --> and <!-- BENCH:<setting> END -->."""
from __future__ import annotations

import importlib.util
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_spec = importlib.util.spec_from_file_location("plot_results", Path(__file__).with_name("plot_results.py"))
assert _spec and _spec.loader
pr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pr)
plt = pr.plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "report" / "figures"
CHAPTER = ROOT / "docs" / "report" / "09_benchmark.md"
SETTINGS = {"d1": "D1: 15 clients, α = 0.5, 3 attackers", "d2": "D2: 20 clients, α = 0.2, 4 attackers"}
ATTACKS = [("clean", "No attack"), ("sign", "Sign flip"), ("label", "Label flip"), ("alie", "ALIE"),
           ("minmax", "Min-Max"), ("backdoor", "Backdoor"), ("stealth", "Stealth backdoor")]
ACC_ATTACKS = ATTACKS[:5]
BACKDOORS = ATTACKS[5:]
# key, label, phase, colour
GRAY, DARK, GUARD = "#9a9893", "#52514e", "#0b7a52"
MODELS = [
    ("fedavg", "FedAvg", "Baseline", GRAY), ("multikrum", "Multi-Krum", "Baseline", GRAY),
    ("fedhep", "FedHEP", "Phase 1", DARK), ("defended", "Defended FedHEP", "Phase 1", DARK),
    ("plus", "FedHEP+", "Phase 3", pr.SLOT[0]), ("drill", "FedHEP+ drill-down", "Phase 4", pr.SLOT[3]),
    ("median", "Median", "Phase 5", pr.SLOT[1]), ("tmean", "Trimmed mean", "Phase 5", pr.SLOT[1]),
    ("bulyan", "Bulyan", "Phase 5", pr.SLOT[1]), ("flame", "FLAME", "Phase 5", pr.SLOT[1]),
    ("fltrust", "FLTrust", "Phase 5", pr.SLOT[1]),
    ("l0", "L0", "Phase 6", pr.SLOT[2]), ("r2", "R2", "Phase 6", pr.SLOT[2]), ("r3", "R3", "Phase 6", pr.SLOT[2]),
    ("guard", "FedHEP-Guard", "Phase 6", GUARD),
]
LABEL = {k: lab for k, lab, _, _ in MODELS}
COLOR = {k: c for k, _, _, c in MODELS}


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def load(outputs: Path) -> dict[tuple[str, str, str], list[dict]]:
    runs: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for f in sorted(outputs.glob("bm_*/metrics.json")):
        m = json.loads(f.read_text())
        if not m.get("final"):
            continue
        name = re.sub(r"_s\d+$", "", f.parent.name)
        setting, attack, model = name.removeprefix("bm_").split("__")
        runs[(setting, attack, model)].append(m)
    return runs


def acc(m: dict) -> float:
    return m["final"]["pooled"]


def asr(m: dict) -> float | None:
    return m["history"][-1].get("asr")


def honest_isolated(m: dict) -> int:
    if not m.get("diagnostics"):
        return 0
    return len(set(m["diagnostics"][-1].get("isolated", [])) - set(m["attackers"]))


def stat(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def models_in(runs: dict, setting: str) -> list[str]:
    present = {k[2] for k in runs if k[0] == setting}
    return [k for k, *_ in MODELS if k in present]


def fig_accuracy(runs: dict, setting: str) -> None:
    """One panel per attack; horizontal bars per model (mean, sd error bar), coloured by phase."""
    models = models_in(runs, setting)
    fig, axes = plt.subplots(1, len(ACC_ATTACKS), figsize=(17, 0.42 * len(models) + 1.6), sharey=True)
    for ax, (attack, title) in zip(axes, ACC_ATTACKS, strict=True):
        for y, model in enumerate(models):
            ms = runs.get((setting, attack, model), [])
            if not ms:
                ax.text(1, y, "not run", va="center", fontsize=7, color=pr.INK2)
                continue
            mu, sd = stat([acc(m) for m in ms])
            ax.barh(y, mu, xerr=sd, color=COLOR[model], height=0.7, capsize=2)
            ax.text(mu + sd + 0.6, y, f"{mu:.1f}", va="center", fontsize=7)
        ax.set_title(title)
        ax.set_xlim(0, 65)
        ax.set_xlabel("accuracy (%)")
    axes[0].set_yticks(range(len(models)), [LABEL[k] for k in models])
    axes[0].invert_yaxis()
    fig.suptitle(f"Accuracy of every model, {SETTINGS[setting]} (mean ± sd over 3 seeds, GPU server)",
                 fontsize=11)
    fig.tight_layout()
    save(fig, FIG / f"bench_{setting}_accuracy.png")


def fig_backdoor(runs: dict, setting: str) -> None:
    """Backdoor success per model for the plain and the stealth backdoor; dots are seeds, hatched bars add the
    depth-exit guard."""
    models = models_in(runs, setting)
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.42 * len(models) + 1.6), sharey=True)
    for ax, (attack, title) in zip(axes, BACKDOORS, strict=True):
        for y, model in enumerate(models):
            ms = [m for m in runs.get((setting, attack, model), []) if asr(m) is not None]
            if not ms:
                ax.text(1, y, "not run" if attack == "backdoor" or model not in ("fedavg", "multikrum")
                        else "needs a FedHEP server", va="center", fontsize=7, color=pr.INK2)
                continue
            vals = [asr(m) for m in ms]
            ax.barh(y, statistics.fmean(vals), color=COLOR[model], height=0.7)
            ax.scatter(vals, [y] * len(vals), s=10, color=pr.INK, zorder=3)
            guarded = [m["history"][-1]["guarded"]["asr"] for m in ms if m["history"][-1].get("guarded")]
            if guarded:
                ax.scatter([statistics.fmean(guarded)], [y], marker="|", s=160, color=pr.INK, zorder=4)
            ax.text(statistics.fmean(vals) + 1.5, y, f"{statistics.fmean(vals):.1f}", va="center", fontsize=7)
        ax.set_title(title)
        ax.set_xlim(0, 100)
        ax.set_xlabel("backdoor success at round 30 (%)  ·  dots: seeds  ·  bar mark: with exit guard")
    axes[0].set_yticks(range(len(models)), [LABEL[k] for k in models])
    axes[0].invert_yaxis()
    fig.suptitle(f"Backdoor success of every model, {SETTINGS[setting]} (lower is better)", fontsize=11)
    fig.tight_layout()
    save(fig, FIG / f"bench_{setting}_backdoor.png")


def fig_tradeoff(runs: dict, setting: str) -> None:
    """Accuracy with no attack against the worst accuracy under the four model-poisoning attacks, and against
    backdoor success: one point per model."""
    models = [k for k in models_in(runs, setting) if runs.get((setting, "clean", k))]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2))
    for model in models:
        clean = statistics.fmean(acc(m) for m in runs[(setting, "clean", model)])
        attacked = [statistics.fmean(acc(m) for m in runs[(setting, a, model)])
                    for a, _ in ACC_ATTACKS[1:] if runs.get((setting, a, model))]
        bd = [asr(m) for m in runs.get((setting, "backdoor", model), []) if asr(m) is not None]
        if attacked:
            a1.scatter(min(attacked), clean, s=60, color=COLOR[model], edgecolor=pr.INK, linewidth=0.5)
            a1.annotate(LABEL[model], (min(attacked), clean), fontsize=7.5, xytext=(4, 3),
                        textcoords="offset points")
        if bd:
            a2.scatter(statistics.fmean(bd), clean, s=60, color=COLOR[model], edgecolor=pr.INK, linewidth=0.5)
            a2.annotate(LABEL[model], (statistics.fmean(bd), clean), fontsize=7.5, xytext=(4, 3),
                        textcoords="offset points")
    a1.set_xlabel("worst accuracy under sign flip, label flip, ALIE, Min-Max (%)  →  better")
    a2.set_xlabel("backdoor success (%)  ←  better")
    for ax in (a1, a2):
        ax.set_ylabel("accuracy with no attack (%)")
    a1.set_title("Robustness against cost")
    a2.set_title("Backdoor against cost")
    fig.suptitle(f"Trade-offs, {SETTINGS[setting]}: the best models sit top-right (left) and top-left (right)",
                 fontsize=11)
    fig.tight_layout()
    save(fig, FIG / f"bench_{setting}_tradeoff.png")


def fig_cost(runs: dict, setting: str) -> None:
    """Honest clients isolated with no attack, and server time per round, per model."""
    models = [k for k in models_in(runs, setting) if runs.get((setting, "clean", k))]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 0.42 * len(models) + 1.6), sharey=True)
    for y, model in enumerate(models):
        ms = runs[(setting, "clean", model)]
        iso = statistics.fmean(honest_isolated(m) for m in ms)
        sec = statistics.fmean(statistics.fmean(m["resources"]["seconds_per_round"]) for m in ms)
        a1.barh(y, iso, color=COLOR[model], height=0.7)
        a1.text(iso + 0.05, y, f"{iso:.1f}", va="center", fontsize=7)
        a2.barh(y, sec, color=COLOR[model], height=0.7)
        a2.text(sec + 0.05, y, f"{sec:.1f}", va="center", fontsize=7)
    a1.set_title("Honest clients isolated per run, no attack")
    a1.set_xlabel("clients")
    a2.set_title("Time per round (training and server, one GPU)")
    a2.set_xlabel("seconds")
    a1.set_yticks(range(len(models)), [LABEL[k] for k in models])
    a1.invert_yaxis()
    fig.suptitle(f"What each defense costs, {SETTINGS[setting]}", fontsize=11)
    fig.tight_layout()
    save(fig, FIG / f"bench_{setting}_cost.png")


def table(runs: dict, setting: str) -> str:
    models = models_in(runs, setting)
    head = ("| Model | Phase | " + " | ".join(t for _, t in ACC_ATTACKS) + " | Backdoor success | Stealth success "
            "| Honest isolated, no attack |")
    rows = [head, "|" + "---|" * (len(ACC_ATTACKS) + 5)]
    phase = {k: p for k, _, p, _ in MODELS}
    for model in models:
        cells = []
        for attack, _ in ACC_ATTACKS:
            ms = runs.get((setting, attack, model), [])
            if not ms:
                cells.append("—")
                continue
            mu, sd = stat([acc(m) for m in ms])
            cells.append(f"{mu:.2f} ± {sd:.2f}" + ("" if len(ms) == 3 else f" (n={len(ms)})"))
        for attack, _ in BACKDOORS:
            vals = [asr(m) for m in runs.get((setting, attack, model), []) if asr(m) is not None]
            cells.append(f"{statistics.fmean(vals):.1f}%" + ("" if len(vals) == 3 else f" (n={len(vals)})")
                         if vals else "—")
        clean = runs.get((setting, "clean", model), [])
        cells.append(f"{statistics.fmean(honest_isolated(m) for m in clean):.1f}" if clean else "—")
        name = f"**{LABEL[model]}**" if model == "guard" else LABEL[model]
        rows.append(f"| {name} | {phase[model]} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


FIGURES = [("accuracy", "Accuracy of every model"), ("backdoor", "Backdoor success of every model"),
           ("tradeoff", "Trade-offs"), ("cost", "Cost of every defense")]


def figure_block() -> str:
    lines = []
    for setting, title in SETTINGS.items():
        present = [(k, t) for k, t in FIGURES if (FIG / f"bench_{setting}_{k}.png").exists()]
        if present:
            lines.append(f"**{title}**\n")
            lines += [f"![{t}, {setting.upper()}](figures/bench_{setting}_{k}.png)\n" for k, t in present]
    return "\n".join(lines) if lines else "Pending."


def replace_block(text: str, key: str, body: str) -> str:
    start, end = f"<!-- BENCH:{key} START -->", f"<!-- BENCH:{key} END -->"
    if start not in text or end not in text:
        return text
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    return f"{before}{start}\n{body}\n{end}{after}"


def write_tables(runs: dict) -> None:
    if not CHAPTER.exists():
        return
    text = replace_block(CHAPTER.read_text(), "figures", figure_block())
    for setting in SETTINGS:
        if models_in(runs, setting):
            text = replace_block(text, setting, table(runs, setting))
    CHAPTER.write_text(text)
    print(f"updated tables in {CHAPTER.relative_to(ROOT)}")


def main() -> None:
    pr.style()
    runs = load(Path(sys.argv[1] if len(sys.argv) > 1 else "outputs"))
    if not runs:
        print("no benchmark runs (bm_*) found")
        return
    for setting in SETTINGS:
        if not models_in(runs, setting):
            continue
        fig_accuracy(runs, setting)
        fig_backdoor(runs, setting)
        fig_tradeoff(runs, setting)
        fig_cost(runs, setting)
    write_tables(runs)


if __name__ == "__main__":
    main()
