"""Charts for docs/EXPERIMENT_LOG.md, drawn from finished runs (outputs/<name>_s<seed>/metrics.json).

  python scripts/plot_results.py outputs            # writes docs/figures/*.png

Numbers for the authors' code and the paper are not in outputs/; they are the values recorded in the experiment
log (REFERENCE below). A figure whose runs are missing is skipped with a message."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "docs" / "figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical order (blue, orange, aqua, yellow)
METHODS = [("fedavg", "FedAvg"), ("multikrum", "Multi-Krum"), ("fedhep", "FedHEP"),
           ("defended", "Defended FedHEP")]
ATTACKS = [("clean", "No attack"), ("sign_flip", "Sign flip"), ("label_flip", "Label flip"),
           ("noise", "Random noise"), ("backdoor", "Backdoor"), ("backdoor_x3", "Backdoor ×3")]

# (mean, sd or None) recorded in EXPERIMENT_LOG.md: authors' code run on your Mac, and the paper's tables.
REFERENCE = {
    "FedAvg · IID": {"runs": "cifar100_fedavg", "authors": (38.04, 0.94), "paper": (41.46, 0.36)},
    "FedAvg · α=0.05": {"runs": "cifar100_fedavg_a0.05", "authors": (29.07, 2.05), "paper": (36.29, None)},
    "FedHEP · α=0.05": {"runs": "cifar100_fedhep_a0.05", "authors": (65.68, 0.57), "paper": (65.49, 0.61)},
    "Defended · clean": {"runs": "cifar100_fedhep_defended", "authors": None, "paper": (47.88, 0.46)},
    "Defended · sign flip 20%": {"runs": "cifar100_def_sign_flip_q20", "authors": None, "paper": (40.22, 3.24)},
    "Defended · noise 30%": {"runs": "cifar100_def_noise_q30", "authors": None, "paper": (44.53, 1.86)},
}


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
        "lines.linewidth": 2, "lines.markersize": 5,
    })


def load(outputs: Path) -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(outputs.glob("*/metrics.json")):
        m = json.loads(f.read_text())
        runs[m["name"]].append(m)
    return runs


def mean_sd(xs: list[float]) -> tuple[float, float]:
    return statistics.fmean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def save(fig: plt.Figure, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote docs/figures/{name}")


def bar_labels(ax: plt.Axes, bars, fmt: str = "{:.1f}", errors: list[float] | None = None,
               size: float = 7.5) -> None:
    """Value above each bar, clear of its error bar."""
    for k, b in enumerate(bars):
        h = b.get_height()
        top = h + (errors[k] if errors else 0.0)
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, top), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=size, color=INK2)


# ----------------------------------------------------------------------------- Phase 1


def fig_parity(runs: dict) -> None:
    cases = [(k, v) for k, v in REFERENCE.items() if runs.get(v["runs"])]
    if not cases:
        return print("skip parity: no Phase 1 runs")
    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.26
    for j, (who, label) in enumerate((("ours", "RFL Lab (your Mac)"), ("authors", "Authors' code (your Mac)"),
                                      ("paper", "Paper"))):
        xs, ys, es = [], [], []
        for i, (_, ref) in enumerate(cases):
            val = mean_sd([m["final"]["pooled"] for m in runs[ref["runs"]]]) if who == "ours" else ref[who]
            if val is None:
                continue
            xs.append(i + (j - 1) * width)
            ys.append(val[0])
            es.append(val[1] or 0.0)
        bars = ax.bar(xs, ys, width * 0.92, yerr=es, color=SLOT[j], label=label, capsize=2,
                      error_kw={"elinewidth": 1, "ecolor": INK2})
        bar_labels(ax, bars, errors=es)
    ax.set_xticks(range(len(cases)), [k.replace(" · ", "\n") for k, _ in cases], fontsize=9)
    ax.set_ylabel("Accuracy (%) · mean ± sd over 3 seeds")
    ax.set_title("Phase 1 · RFL Lab reproduces the authors' code (missing bar = not run)")
    ax.legend(loc="upper left", ncols=3)
    ax.set_ylim(0, 78)
    save(fig, "phase1_parity.png")


def fig_parent_agreement(runs: dict) -> None:
    ms = runs.get("cifar100_fedhep_a0.05")
    if not ms or not ms[0].get("diagnostics"):
        return print("skip parent agreement: no FedHEP diagnostics")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for i, m in enumerate(sorted(ms, key=lambda m: m["seed"])):
        d = m["diagnostics"]
        ax.plot([x["round"] for x in d], [x["member_agreement"] for x in d], color=SLOT[i], marker="o",
                markersize=3, label=f"seed {m['seed']}")
    ax.axhline(0.5, color=INK2, linestyle="--", linewidth=1)
    ax.annotate("above 0.5 → parents copy the global model", (25, 0.5), xytext=(0, 5), textcoords="offset points",
                ha="right", fontsize=8, color=INK2)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean member–group cosine")
    ax.set_title("FedHEP · group members always 'agree', so the parent tier never averages (3 seeds overlap)")
    ax.legend(loc="lower right", ncols=3)
    save(fig, "fedhep_parent_agreement.png")


def fig_counter() -> None:
    """Output of scripts/diag_update_norms.py (deterministic, CPU), recorded in the experiment log."""
    norms = {"Weights": (5.2, 6.3), "BatchNorm statistics": (24, 25), "BatchNorm batch counters": (99, 113)}
    cos = {"Full vector (authors)": [0.984, 0.985, 0.984], "Without counters": [0.73, 0.68, 0.72],
           "Weights only": [0.05, 0.05, 0.17]}
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.2))
    bars = a.barh(list(norms), [hi for _, hi in norms.values()], color=SLOT[0], height=0.5)
    for bar, (lo, hi) in zip(bars, norms.values(), strict=True):
        a.annotate(f"{lo}–{hi}", (hi, bar.get_y() + bar.get_height() / 2), xytext=(4, 0),
                   textcoords="offset points", va="center", fontsize=8, color=INK2)
    a.set_xlabel("Norm of one client's update (range over 3 clients)")
    a.set_title("What a client's update is made of")
    a.invert_yaxis()
    for i, v in enumerate(cos.values()):
        b.scatter(v, [i] * 3, color=SLOT[1], s=40, zorder=3)
        b.annotate(f"{statistics.fmean(v):.2f}", (max(v), i), xytext=(8, 0), textcoords="offset points",
                   va="center", fontsize=8, color=INK2)
    b.set_yticks(range(len(cos)), list(cos))
    b.set_xlim(0, 1.1)
    b.set_ylim(len(cos) - 0.5, -0.5)
    b.set_xlabel("Cosine between clients with disjoint classes (3 pairs)")
    b.set_title("Similarity the grouping sees")
    save(fig, "batchnorm_counter_dominance.png")


def _share_series(m: dict) -> tuple[list[int], list[float], list[float]]:
    att = set(m["attackers"])
    rounds, bad, good = [], [], []
    for d in m["diagnostics"]:
        n = len(d["global_trust"])
        shares = {int(c): t * n for c, t in d["global_trust"].items()}
        a = [s for c, s in shares.items() if c in att]
        h = [s for c, s in shares.items() if c not in att]
        if a and h:
            rounds.append(d["round"])
            bad.append(statistics.fmean(a))
            good.append(statistics.fmean(h))
    return rounds, bad, good


def fig_trust(runs: dict, prefix: str, cases: list[tuple[str, str]], name: str, title: str) -> None:
    cases = [(r, t) for r, t in cases if runs.get(prefix + r)]
    if not cases:
        return print(f"skip {name}: no runs")
    cols = min(2, len(cases))
    rows = -(-len(cases) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 3 * rows + 0.6), sharey=True, squeeze=False)
    for ax, (run, label) in zip(axes.flat, cases, strict=True):
        for m in runs[prefix + run]:
            r, bad, good = _share_series(m)
            ax.plot(r, good, color=SLOT[0], alpha=0.85, linewidth=1.5)
            ax.plot(r, bad, color=SLOT[1], alpha=0.85, linewidth=1.5)
        ax.axhline(1.0, color=INK2, linestyle="--", linewidth=1)
        ax.set_title(label)
        ax.set_xlabel("Round")
    for ax in axes[:, 0]:
        ax.set_ylabel("Global weight (1 = equal share)")
    handles = [plt.Line2D([], [], color=SLOT[0]), plt.Line2D([], [], color=SLOT[1])]
    fig.legend(handles, ["honest clients (mean, one line per seed)", "attackers (mean, one line per seed)"],
               loc="upper right", ncols=2, fontsize=8.5)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93 if rows == 1 else 0.95))
    save(fig, name)


# ----------------------------------------------------------------------------- Phase 2


def _p2(runs: dict, method: str, attack: str) -> list[dict]:
    return runs.get(f"p2__{method}__{attack}", [])


def fig_p2_accuracy(runs: dict) -> None:
    if not _p2(runs, "fedavg", "clean"):
        return print("skip Phase 2: no runs")
    fig, ax = plt.subplots(figsize=(11, 4))
    width = 0.2
    for j, (meth, label) in enumerate(METHODS):
        xs, ys, es = [], [], []
        for i, (att, _) in enumerate(ATTACKS):
            ms = _p2(runs, meth, att)
            if ms:
                mu, sd = mean_sd([m["final"]["pooled"] for m in ms])
                xs.append(i + (j - 1.5) * width)
                ys.append(mu)
                es.append(sd)
        bars = ax.bar(xs, ys, width * 0.92, yerr=es, color=SLOT[j], label=label, capsize=2,
                      error_kw={"elinewidth": 1, "ecolor": INK2})
        bar_labels(ax, bars, errors=es)
    ax.set_xticks(range(len(ATTACKS)), [t for _, t in ATTACKS])
    ax.set_ylabel("Accuracy of honest clients (%)")
    ax.set_title("Phase 2 · accuracy under attack (3 of 15 clients attack · α=0.5 · 30 rounds · 3 seeds)")
    ax.legend(loc="upper right", ncols=4)
    ax.set_ylim(0, 60)
    save(fig, "phase2_accuracy.png")


def fig_p2_asr(runs: dict) -> None:
    cases = [("backdoor", "Backdoor"), ("backdoor_x3", "Backdoor ×3 (model replacement)")]
    if not _p2(runs, "fedavg", "backdoor"):
        return
    fig, ax = plt.subplots(figsize=(8, 3.4))
    width = 0.2
    for j, (meth, label) in enumerate(METHODS):
        xs, ys, es = [], [], []
        for i, (att, _) in enumerate(cases):
            last = [m["history"][-1] for m in _p2(runs, meth, att)]
            vals = [h["asr"] for h in last if h.get("asr") is not None]
            if vals:
                mu, sd = mean_sd(vals)
                xs.append(i + (j - 1.5) * width)
                ys.append(mu)
                es.append(sd)
        bars = ax.bar(xs, ys, width * 0.92, yerr=es, color=SLOT[j], label=label, capsize=2,
                      error_kw={"elinewidth": 1, "ecolor": INK2})
        bar_labels(ax, bars, errors=es)
    ax.set_xticks(range(len(cases)), [t for _, t in cases])
    ax.set_ylabel("Attack success rate (%) · lower is better")
    ax.set_title("Phase 2 · backdoor success on honest clients' test images")
    ax.legend(loc="upper left", ncols=4, fontsize=8.5)
    ax.set_ylim(0, 110)
    save(fig, "phase2_backdoor_asr.png")


def _round_mean(ms: list[dict], key: str) -> float | None:
    per_seed = []
    for m in ms:
        vals = [d[key] for d in m.get("detection", []) if d.get(key) is not None]
        if vals:
            per_seed.append(statistics.fmean(vals))
    return statistics.fmean(per_seed) if per_seed else None


def fig_p2_detection(runs: dict) -> None:
    attacks = [a for a in ATTACKS if a[0] != "clean"]
    if not _p2(runs, "multikrum", "sign_flip"):
        return
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6), sharex=True)
    width = 0.36
    for j, (meth, label) in enumerate([METHODS[1], METHODS[3]]):
        for ax, key in ((a, "auroc"), (b, "f1")):
            xs, ys = [], []
            for i, (att, _) in enumerate(attacks):
                v = _round_mean(_p2(runs, meth, att), key)
                if v is not None:
                    xs.append(i + (j - 0.5) * width)
                    ys.append(v)
            bars = ax.bar(xs, ys, width * 0.92, color=SLOT[1 if j == 0 else 3], label=label)
            bar_labels(ax, bars, "{:.2f}")
    a.axhline(0.5, color=INK2, linestyle="--", linewidth=1)
    a.annotate("chance", (-0.45, 0.5), xytext=(0, 3), textcoords="offset points", ha="left", fontsize=8,
               color=INK2)
    a.set_title("Do trust scores rank attackers below honest clients?")
    a.set_ylabel("AUROC (mean over rounds)")
    b.set_title("Do they exclude attackers, and only attackers?")
    b.set_ylabel("F1 of excluded clients (mean over rounds)")
    for ax in (a, b):
        ax.set_ylim(0, 1.12)
        ax.set_xticks(range(len(attacks)), [t for _, t in attacks], fontsize=8.5)
    fig.legend(*a.get_legend_handles_labels(), loc="upper right", ncols=2, fontsize=9)
    fig.suptitle("Phase 2 · detection (only Multi-Krum and Defended FedHEP have a detector)", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fig.text(0.01, 0.885, "Multi-Krum always excludes 12 of 15 clients: all 3 attackers, but 9 honest ones too "
             "(precision 0.25). Defended FedHEP almost never excludes anyone.", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save(fig, "phase2_detection.png")


def fig_p2_curves(runs: dict) -> None:
    if not _p2(runs, "fedavg", "clean"):
        return
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)
    for ax, (att, title) in zip(axes.flat, ATTACKS, strict=True):
        for j, (meth, label) in enumerate(METHODS):
            ms = _p2(runs, meth, att)
            if not ms:
                continue
            rounds = [h["round"] for h in ms[0]["history"]]
            ys = [statistics.fmean(m["history"][k]["pooled_acc"] for m in ms) for k in range(len(rounds))]
            ax.plot(rounds, ys, color=SLOT[j], marker="o", label=label)
        ax.set_title(title)
    for ax in axes[1]:
        ax.set_xlabel("Round")
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy (%)")
    axes[0][0].legend(loc="upper left", fontsize=8.5)
    fig.suptitle("Phase 2 · accuracy over rounds (mean of 3 seeds)", x=0.01, ha="left", fontweight="bold",
                 fontsize=11)
    fig.tight_layout()
    save(fig, "phase2_curves.png")


# ----------------------------------------------------------------------------- Phase 2 diagnosis


def fig_multikrum_sizes(runs: dict) -> None:
    ms = [m for att in ("clean", "label_flip", "backdoor") for m in _p2(runs, "multikrum", att)]
    if not ms:
        return
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for i, att in enumerate(("clean", "label_flip", "backdoor")):
        xs, ys = [], []
        for m in _p2(runs, "multikrum", att):
            sz, n = m["client_train_sizes"], len(m["client_train_sizes"])
            med = statistics.median(sz)
            kept = defaultdict(int)
            for d in m["detection"]:
                for c in set(range(n)) - set(d["flagged"]):
                    kept[c] += 1
            for c in range(n):
                if c not in m["attackers"]:
                    xs.append(sz[c] - med)
                    ys.append(kept[c])
        ax.scatter(xs, ys, s=22, color=SLOT[i], alpha=0.8, label=dict(ATTACKS)[att], zorder=3)
    ax.set_xlabel("Client's training set size minus the median client's (images)")
    ax.set_ylabel("Rounds kept (of 30)")
    ax.set_title("Multi-Krum keeps the median-sized clients (honest clients, 3 seeds)")
    ax.legend(loc="upper right", fontsize=8.5)
    save(fig, "multikrum_kept_vs_size.png")


def fig_defended_failure(runs: dict) -> None:
    ms = {att: _p2(runs, "defended", att) for att, _ in ATTACKS if att != "backdoor_x3"}
    if not ms["clean"]:
        return
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.4))
    rounds = range(1, 31)
    uniform = [0] * 30
    total = 0
    for att_runs in ms.values():
        for m in att_runs:
            total += 1
            for d in m["diagnostics"]:
                n = len(d["global_trust"])
                if n and max(abs(t * n - 1) for t in d["global_trust"].values()) < 1e-5:
                    uniform[d["round"] - 1] += 1
    a.bar(list(rounds), [u / total for u in uniform], color=SLOT[0], width=0.8)
    a.set_xlabel("Round")
    a.set_ylabel(f"Share of runs (of {total})")
    a.set_title("Rounds where Global weights everyone equally")
    a.set_ylim(0, 1.05)
    for j, att in enumerate(("clean", "sign_flip")):
        spreads = defaultdict(list)
        for m in ms[att]:
            bad = set(m["attackers"])
            for d in m["diagnostics"]:
                n = len(d["global_trust"])
                sh = [t * n for c, t in d["global_trust"].items() if int(c) not in bad]
                if sh:
                    spreads[d["round"]].append(max(sh) - min(sh))
        b.plot(sorted(spreads), [statistics.fmean(spreads[r]) for r in sorted(spreads)], color=SLOT[j],
               label=dict(ATTACKS)[att])
    b.set_xlabel("Round")
    b.set_ylabel("Max - min weight among honest clients")
    b.set_title("Honest clients drift apart as training goes on")
    b.legend(loc="upper left", fontsize=8.5)
    fig.suptitle("Defended FedHEP · why the defense fades (Phase 2 runs, 3 seeds)", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, "defended_failure_diagnosis.png")


# ----------------------------------------------------------------------------- Phase 3

def fig_variants(runs: dict, rows_spec: list[tuple[str, str]], seeds: set[int], name: str, title: str) -> None:
    """Accuracy and detection F1 per attack for several methods, mean (± sd when several seeds) over `seeds`.
    rows_spec: (label, run-name prefix); run names are f"{prefix}__{attack}"."""
    def get(prefix: str, att: str) -> list[dict]:
        return [m for m in runs.get(f"{prefix}__{att}", []) if m["seed"] in seeds]

    def complete(prefix: str) -> bool:  # every attack it has, with every seed asked for
        counts = [len(get(prefix, a)) for a, _ in ATTACKS]
        return any(counts) and all(c in (0, len(seeds)) for c in counts)

    rows = [(label, prefix) for label, prefix in rows_spec if complete(prefix)]
    if len(rows) < 2:
        return print(f"skip {name}: no runs")
    attacks = [a for a in ATTACKS if a[0] != "backdoor_x3"]
    palette = [INK2] + SLOT
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    width = 0.8 / len(rows)
    for j, (label, prefix) in enumerate(rows):
        xs, acc, err, fx, f1 = [], [], [], [], []
        for i, (att, _) in enumerate(attacks):
            ms = get(prefix, att)
            if not ms:
                continue
            x = i + (j - (len(rows) - 1) / 2) * width
            mu, sd = mean_sd([m["final"]["pooled"] for m in ms])
            xs.append(x)
            acc.append(mu)
            err.append(sd)
            v = _round_mean(ms, "f1")
            if v is not None:  # undefined without attackers
                fx.append(x)
                f1.append(v)
        bars = ax.bar(xs, acc, width * 0.92, yerr=err if len(seeds) > 1 else None, color=palette[j], label=label,
                      capsize=2, error_kw={"elinewidth": 1, "ecolor": INK2})
        bar_labels(ax, bars, errors=err if len(seeds) > 1 else None)
        bars = bx.bar(fx, f1, width * 0.92, color=palette[j])
        bar_labels(bx, bars, "{:.2f}")
    ax.set_ylabel("Accuracy of honest clients (%)")
    ax.set_ylim(0, 60)
    ax.set_title("Accuracy")
    bx.set_ylabel("F1 of excluded clients")
    bx.set_ylim(0, 1.12)
    bx.set_title("Detection (1 = exactly the attackers excluded)")
    bx.set_xticks(range(len(attacks)), [t for _, t in attacks])
    fig.legend(*ax.get_legend_handles_labels(), loc="upper right", ncols=min(3, len(rows)), fontsize=8.5)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, name)


def _p3_events(m: dict) -> dict[str, object]:
    """Per run: readmissions (round, attacker?), rounds in which attackers were aggregated, honest clients ever
    isolated. A readmission is a client leaving the isolated set; 'excluded' is what the round left out after
    the cap (older runs log it under detection 'flagged')."""
    bad, prev, ever = set(m["attackers"]), set(), set()
    readmit, in_model = [], 0
    for i, d in enumerate(m["diagnostics"]):
        iso = set(d["isolated"])
        excluded = set(d.get("excluded", m["detection"][i]["flagged"]))
        readmit += [(d["round"], c in bad) for c in prev - iso]
        in_model += len(bad - excluded)
        ever |= iso
        prev = iso
    return {"readmit": readmit, "in_model": in_model, "honest_ever": len(ever - bad)}


def fig_phase3_variants(runs: dict) -> None:
    """Phase 3b/3c variants explained: who readmission brings back and when, how many attacker-rounds each
    variant lets into the model, and what the full-population yardstick costs in honest isolations."""
    arms = [("p3__zmr", "FedHEP+"), ("p3b__zmr_ra", "+ readmit"), ("p3b__zm_ra_cap", "+ readmit + cap\n(no regroup)"),
            ("p3b__zmr_ra_cap", "+ readmit + cap"), ("p3c__zmp", "full population\n(no regroup)"),
            ("p3c__zmrp", "full population")]
    attacks = ["clean", "sign_flip", "label_flip", "noise", "backdoor"]
    ev = {(a, t): [_p3_events(m) for m in runs.get(f"{a}__{t}", []) if m["seed"] in (42, 123, 7)]
          for a, _ in arms for t in attacks}
    if not all(len(v) == 3 for v in ev.values()):
        return print("skip phase3 variants: runs missing")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw={"width_ratios": [1.1, 1.3, 1]})
    ax = axes[0]
    rounds = range(1, 31)
    back = [r for t in attacks for e in ev[("p3b__zmr_ra", t)] for r in e["readmit"]]
    att = [sum(1 for rr, a in back if rr == r and a) for r in rounds]
    hon = [sum(1 for rr, a in back if rr == r and not a) for r in rounds]
    ax.bar(rounds, att, color=SLOT[1], label=f"attackers ({sum(att)})")
    ax.bar(rounds, hon, bottom=att, color="#9a9893", label=f"honest clients ({sum(hon)})")
    ax.set_xlabel("round")
    ax.set_ylabel("clients readmitted")
    ax.set_title("Readmission: who comes back, and when\n(+ readmit, all 5 conditions, 3 seeds)", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    shown = [t for t in attacks if t != "clean"]
    w = 0.8 / 4
    for j, (a, label) in enumerate(arms[:4]):
        vals = [sum(e["in_model"] for e in ev[(a, t)]) / 3 for t in shown]
        bars = ax.bar([i + (j - 1.5) * w for i in range(len(shown))], vals, w - 0.02, color=[INK2, *SLOT][j],
                      label=label.replace("\n", " "))
        bar_labels(ax, bars, "{:.0f}", size=7)
    ax.set_xticks(range(len(shown)), ["Sign flip", "Label flip", "Noise", "Backdoor"])
    ax.set_ylabel("attacker-rounds aggregated per run (of 90)")
    ax.set_title("Attackers let back into the model", fontsize=10)
    ax.legend(fontsize=7.5)
    ax = axes[2]
    picks = [arms[0], arms[4], arms[5]]
    hon_iso = [sum(e["honest_ever"] for t in attacks for e in ev[(a, t)]) for a, _ in picks]
    asr = [mean_sd([m["history"][-1].get("asr") or 0.0 for m in runs[f"{a}__backdoor"] if m["seed"] in (42, 123, 7)])[0]
           for a, _ in picks]
    bars = ax.bar(range(3), hon_iso, 0.55, color=[INK2, SLOT[3], SLOT[3]])   # its own colour: not the arms of panel 2
    bar_labels(ax, bars, "{:.0f}", size=8)
    for i, v in enumerate(asr):
        ax.text(i, hon_iso[i] / 2, f"backdoor\n{v:.0f}%", ha="center", va="center", fontsize=8,
                color="white" if i == 0 else INK)
    ax.set_xticks(range(3), [label for _, label in picks], fontsize=8.5)
    ax.set_ylabel("honest clients isolated (15 runs)")
    ax.set_title("Full-population yardstick", fontsize=10)
    fig.suptitle("Phase 3b and 3c variants of FedHEP+ (D1, 3 attackers, Mac, seeds 42, 123, 7)", fontsize=11)
    fig.tight_layout()
    save(fig, "phase3_variants_explained.png")


def fig_cascade(runs: dict) -> None:
    cases = [("p3__z__clean", "robust z, no attack"), ("p3__zm__sign_flip", "z + median, sign flip"),
             ("p3__zmr__noise", "z + median + regroup, noise"),
             ("p3__zmr__sign_flip", "z + median + regroup, sign flip")]
    cases = [(n, t) for n, t in cases if runs.get(n)]
    if not cases:
        return print("skip cascade: no runs")
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for j, (name, label) in enumerate(cases):
        m = [m for m in runs[name] if m["seed"] == 42][0]
        bad = set(m["attackers"])
        d = m["diagnostics"]
        ax.plot([x["round"] for x in d], [len(set(x["isolated"]) - bad) for x in d], color=SLOT[j], label=label)
    ax.set_xlabel("Round")
    ax.set_ylabel("Honest clients isolated")
    ax.set_title("Phase 3 · honest clients isolated one after another (seed 42)")
    ax.legend(loc="upper left", fontsize=8.5)
    save(fig, "phase3_cascade.png")


def fig_backdoor_variants(runs: dict, rows_spec: list[tuple[str, str]], name: str, title: str) -> None:
    rows = [(t, runs.get(f"{p}__backdoor", [])) for t, p in rows_spec]
    rows = [(t, ms) for t, ms in rows if len(ms) == 3]
    if len(rows) < 2:
        return print(f"skip {name}: no runs")
    fig, ax = plt.subplots(figsize=(8, 3.4))
    palette = [INK2] + SLOT
    for j, (_, ms) in enumerate(rows):
        vals = [m["history"][-1]["asr"] for m in ms]
        bar = ax.bar([j], [statistics.fmean(vals)], 0.6, color=palette[j])
        ax.scatter([j] * len(vals), vals, color=INK, s=14, zorder=3)
        bar_labels(ax, bar, errors=[max(vals) - statistics.fmean(vals)])
    ax.set_xticks(range(len(rows)), [t.replace(" + ", "\n+ ") for t, _ in rows], fontsize=8.5)
    ax.set_ylabel("Backdoor success (%) · lower is better")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    save(fig, name)


def fig_p4(runs: dict) -> None:
    cases = [("sign_flip", "Sign flip"), ("label_flip", "Label flip"), ("backdoor", "Backdoor")]
    places = [("conc", "concentrated"), ("spread", "spread")]
    if not any(n.startswith("p4__") for n in runs):
        return print("skip Phase 4: no runs")
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9), gridspec_kw={"width_ratios": [1.5, 0.7, 1.2]})
    labels = [f"{t}\n{p}" for _, t in cases for _, p in places]
    methods = (("p4__base", "FedHEP+ (screening)"), ("p4__drill", "FedHEP+ with drill-down"))
    for j, (meth, label) in enumerate(methods):
        acc, asr, pf1, xs = [], [], [], []
        for i, (att, _) in enumerate(cases):
            for k, (pl, _) in enumerate(places):
                ms = runs.get(f"{meth}__{att}_{pl}", [])
                x = (2 * i + k) + (j - 0.5) * 0.38
                if ms:
                    xs.append(x)
                    acc.append(statistics.fmean(m["final"]["pooled"] for m in ms))
                    if att == "backdoor":
                        asr.append((k + (j - 0.5) * 0.38, statistics.fmean(m["history"][-1]["asr"] for m in ms)))
                    vals = [d["group"]["f1"] for m in ms for d in m["detection"]
                            if d.get("group", {}).get("f1") is not None]
                    pf1.append((x, statistics.fmean(vals) if vals else None))
        bar_labels(axes[0], axes[0].bar(xs, acc, 0.36, color=SLOT[j], label=label), size=6.5)
        if asr:
            bar_labels(axes[1], axes[1].bar([a for a, _ in asr], [b for _, b in asr], 0.36, color=SLOT[j]))
        pts = [(x, v) for x, v in pf1 if v is not None]
        if pts:
            bar_labels(axes[2], axes[2].bar([x for x, _ in pts], [v for _, v in pts], 0.36, color=SLOT[j]),
                       "{:.2f}")
    axes[0].set_xticks(range(6), labels, fontsize=8)
    axes[0].set_ylim(0, 60)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Accuracy")
    axes[1].set_xticks(range(2), [p for _, p in places])
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Backdoor success (%)")
    axes[1].set_title("Backdoor success · lower is better")
    axes[2].set_xticks(range(6), labels, fontsize=8)
    axes[2].set_ylim(0, 1.12)
    axes[2].set_ylabel("F1 of flagged parents")
    axes[2].set_title("Parent-level detection (drill-down only)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncols=2, fontsize=9)
    fig.suptitle("Phase 4 · drill-down vs screening alone (mean over seeds)", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "phase4_drilldown.png")


def fig_p4_paired(runs: dict) -> None:
    """Drill-down minus screening alone, seed by seed (same seed = same data, attackers and groups), and why the
    concentrated backdoor got worse: attackers' early z."""
    settings = [("clean", "No attack"), ("sign_flip_conc", "Sign flip\nconc."),
                ("sign_flip_spread", "Sign flip\nspread"),
                ("label_flip_conc", "Label flip\nconc."), ("label_flip_spread", "Label flip\nspread"),
                ("backdoor_conc", "Backdoor\nconc."), ("backdoor_spread", "Backdoor\nspread")]
    if not runs.get("p4__drill__backdoor_conc"):
        return print("skip Phase 4 paired: no runs")

    def paired(setting: str, value) -> list[float]:
        base = {m["seed"]: m for m in runs.get(f"p4__base__{setting}", [])}
        drill = runs.get(f"p4__drill__{setting}", [])
        return [value(m) - value(base[m["seed"]]) for m in drill if m["seed"] in base]

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9), gridspec_kw={"width_ratios": [1.7, 0.6, 0.9]})
    for ax, cases, value, sign, ylabel, title in (
            (axes[0], settings, lambda m: m["final"]["pooled"], 1, "Δ accuracy (points)",
             "Accuracy · drill-down minus screening alone"),
            (axes[1], settings[5:], lambda m: m["history"][-1]["asr"], -1, "Δ backdoor success (points)",
             "Backdoor success · lower is better")):
        for i, (setting, _) in enumerate(cases):
            ds = paired(setting, value)
            if ds:
                mu = statistics.fmean(ds)
                ax.bar(i, mu, 0.55, color=SLOT[0] if sign * mu >= 0 else SLOT[1], alpha=0.85)  # blue = better
                ax.scatter([i] * len(ds), ds, color=INK, s=14, zorder=3)
                ax.annotate(f"{mu:+.2f}", (i, mu), xytext=(16, -3), textcoords="offset points", fontsize=8,
                            color=INK2)
        ax.axhline(0, color=INK2, linewidth=1)
        ax.set_xticks(range(len(cases)), [label for _, label in cases], fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[0].axhspan(-1, 1, color=GRID, alpha=0.5, zorder=0)
    axes[0].text(len(settings) - 0.5, 0.9, "seed spread (D23)", ha="right", va="top", fontsize=7.5, color=INK2)

    ax = axes[2]
    for j, (meth, label) in enumerate((("p4__base", "screening alone"), ("p4__drill", "with drill-down"))):
        for k, m in enumerate(sorted(runs.get(f"{meth}__backdoor_conc", []), key=lambda m: m["seed"])):
            att = set(m["attackers"])
            zs = [v for d in m["diagnostics"][:10] for c, v in d["z"].items() if int(c) in att]
            ax.bar(k + (j - 0.5) * 0.38, statistics.fmean(zs), 0.36, color=SLOT[j],
                   label=label if k == 0 else None)
    ax.axhline(-1.5, color=INK2, linewidth=1, linestyle="--")
    ax.text(2.6, -1.47, "isolation cut on evidence", ha="right", va="bottom", fontsize=7.5, color=INK2)
    seeds = sorted(m["seed"] for m in runs["p4__base__backdoor_conc"])
    ax.set_xticks(range(len(seeds)), [f"seed {s}" for s in seeds], fontsize=8)
    ax.set_ylabel("mean z, rounds 1–10")
    ax.set_title("Backdoor conc. · attackers' z")
    ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Phase 4 · paired differences (bar = mean, dots = seeds; blue = drill-down better, "
                 "orange = worse)", x=0.01, ha="left", fontweight="bold",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "phase4_paired.png")


def fig_p4b(runs: dict) -> None:
    """Drill-down v1 and v2 against screening alone (the Phase 4 base runs), same seeds."""
    settings = [("clean", "No attack"), ("label_flip_conc", "Label flip\nconc."),
                ("backdoor_conc", "Backdoor\nconc."), ("backdoor_spread", "Backdoor\nspread")]
    variants = [("p4__drill", "v1: stricter cut, one round"), ("p4b__v2", "v2: member check when flagged"),
                ("p4b__v2_all", "v2: member check, every parent")]
    if not runs.get("p4b__v2__backdoor_conc"):
        return print("skip Phase 4b: no runs")
    base = {s: {m["seed"]: m for m in runs.get(f"p4__base__{s}", [])} for s, _ in settings}

    def honest_isolated(m: dict) -> int:
        return len(set(m["diagnostics"][-1]["isolated"]) - set(m["attackers"]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0), gridspec_kw={"width_ratios": [1.3, 0.8, 1.3]})
    width = 0.26
    for j, (prefix, label) in enumerate(variants):
        for i, (s, _) in enumerate(settings):
            ms = [m for m in runs.get(f"{prefix}__{s}", []) if m["seed"] in base[s]]
            if not ms:
                continue
            x = i + (j - 1) * width
            ds = [m["final"]["pooled"] - base[s][m["seed"]]["final"]["pooled"] for m in ms]
            axes[0].bar(x, statistics.fmean(ds), width * 0.92, color=SLOT[j + 1], label=label if i == 0 else None)
            axes[0].scatter([x] * len(ds), ds, color=INK, s=10, zorder=3)
            hs = [honest_isolated(m) - honest_isolated(base[s][m["seed"]]) for m in ms]
            axes[2].bar(x, sum(hs), width * 0.92, color=SLOT[j + 1])
    for j, (prefix, _) in enumerate([("p4__base", "")] + variants):
        for k, s in enumerate(("backdoor_conc", "backdoor_spread")):
            ms = runs.get(f"{prefix}__{s}", [])
            if ms:
                x = k + (j - 1.5) * 0.2
                vals = [m["history"][-1]["asr"] for m in ms]
                bar_labels(axes[1], axes[1].bar(x, statistics.fmean(vals), 0.18, color=SLOT[j]), size=6.5)
                axes[1].scatter([x] * len(vals), vals, color=INK, s=10, zorder=3)
    axes[0].axhspan(-1, 1, color=GRID, alpha=0.5, zorder=0)
    axes[0].axhline(0, color=INK2, linewidth=1)
    axes[0].set_xticks(range(len(settings)), [t for _, t in settings], fontsize=8)
    axes[0].set_ylabel("Δ accuracy vs screening alone")
    axes[0].set_title("Accuracy (shaded: seed spread, D23)")
    axes[1].set_xticks(range(2), ["concentrated", "spread"])
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Backdoor success (%)")
    axes[1].set_title("Backdoor success · lower is better")
    axes[2].axhline(0, color=INK2, linewidth=1)
    axes[2].set_xticks(range(len(settings)), [t for _, t in settings], fontsize=8)
    axes[2].set_ylabel("Δ honest clients isolated (3 seeds)")
    axes[2].set_title("False isolations · lower is better")
    handles = [plt.Rectangle((0, 0), 1, 1, color=SLOT[0])] + [plt.Rectangle((0, 0), 1, 1, color=SLOT[j + 1])
                                                               for j in range(len(variants))]
    fig.legend(handles, ["screening alone (Phase 4 base)"] + [v for _, v in variants], loc="upper right", ncols=4,
               fontsize=8.5)
    fig.suptitle("Phase 4b · drill-down v2 vs v1 (bar = mean, dots = seeds)", x=0.01, ha="left", fontweight="bold",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save(fig, "phase4b_drilldown.png")


def fig_ceiling(runs: dict) -> None:
    cases = [("FedHEP\nα=0.5", "p2__fedhep__clean", "p3__fedhep_w__clean"),
             ("FedHEP\nα=0.05", "cifar100_fedhep_a0.05", "p3__fedhep_w_a005__clean")]
    if not runs.get("p3__fedhep_w__clean"):
        return print("skip ceiling: no runs")
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6))
    for j, (label, idx) in enumerate((("grouping on full update (authors)", 1), ("grouping on weights only", 2))):
        xs, ys, es = [], [], []
        for i, case in enumerate(cases):
            seeds = {m["seed"] for m in runs.get(cases[i][2], [])}
            ms = [m for m in runs.get(case[idx], []) if m["seed"] in seeds]
            if ms:
                mu, sd = mean_sd([m["final"]["pooled"] for m in ms])
                xs.append(i + (j - 0.5) * 0.36)
                ys.append(mu)
                es.append(sd)
        bars = a.bar(xs, ys, 0.33, color=SLOT[j], label=label)
        bar_labels(a, bars)
    a.set_xticks(range(len(cases)), [c[0] for c in cases])
    a.set_ylim(0, 80)
    a.set_ylabel("Accuracy (%), same seeds")
    a.set_title("FedHEP, no attack: fixing the grouping signal")
    a.legend(loc="upper left", fontsize=8.5)
    longer = (("p2__defended__clean", "30 rounds"), ("p3__defended_50r__clean", "50 rounds"))
    for j, (name, label) in enumerate(longer):
        ms = [m for m in runs.get(name, []) if m["seed"] == 42]
        if ms:
            h = ms[0]["history"]
            b.plot([x["round"] for x in h], [x["pooled_acc"] for x in h], color=SLOT[j], marker="o", label=label)
    b.set_xlabel("Round")
    b.set_ylabel("Accuracy (%)")
    b.set_title("Defended FedHEP, no attack: training longer (seed 42)")
    b.legend(loc="lower right", fontsize=8.5)
    fig.suptitle("Phase 3 · how high can clean accuracy go?", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, "phase3_ceiling.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("outputs")
    runs = load(Path(ap.parse_args().outputs))
    style()
    fig_parity(runs)
    fig_parent_agreement(runs)
    fig_counter()
    fig_trust(runs, "cifar100_def_", [("sign_flip_q20", "Sign flip, 20% (Phase 1c)"),
                                      ("noise_q30", "Random noise, 30% (Phase 1c)")],
              "defended_trust_phase1c.png", "Defended FedHEP · weight Global gives attackers vs honest clients")
    fig_trust(runs, "p2__defended__", [("sign_flip", "Sign flip"), ("label_flip", "Label flip"),
                                       ("noise", "Random noise"), ("backdoor", "Backdoor")],
              "defended_trust_phase2.png", "Defended FedHEP · Global weight by round, 3 attackers (Phase 2)")
    fig_p2_accuracy(runs)
    fig_p2_asr(runs)
    fig_p2_detection(runs)
    fig_p2_curves(runs)
    fig_multikrum_sizes(runs)
    fig_defended_failure(runs)
    fig_variants(runs, [("Defended FedHEP", "p2__defended"), ("robust z", "p3__z"), ("z + median ref.", "p3__zm"),
                        ("z + median, all weights", "p3__zw"), ("z + median + regroup", "p3__zmr")], {42},
                 "phase3_screening.png",
                 "Phase 3 screening · Defended FedHEP vs its fixes (seed 42, same 3 attackers)")
    fig_variants(runs, [("Defended FedHEP", "p2__defended"), ("z + median + regroup", "p3__zmr"),
                        ("z + median + regroup + readmit", "p3b__zmr_ra"),
                        ("z + median + readmit + cap", "p3b__zm_ra_cap"),
                        ("z + median + regroup + readmit + cap", "p3b__zmr_ra_cap")], {42, 123, 7},
                 "phase3b_confirm.png", "Phase 3b · three seeds (mean ± sd)")
    p3b_rows = [("Defended FedHEP", "p2__defended"), ("z + median + regroup", "p3__zmr"),
                ("z + median + regroup + readmit", "p3b__zmr_ra"),
                ("z + median + readmit + cap", "p3b__zm_ra_cap"),
                ("z + median + regroup + readmit + cap", "p3b__zmr_ra_cap")]
    fig_backdoor_variants(runs, p3b_rows, "phase3b_backdoor.png",
                          "Phase 3b · backdoor success, three seeds (bar = mean, dots = seeds)")
    p3c_rows = [("Defended FedHEP", "p2__defended"), ("3b best: z + median + regroup", "p3__zmr"),
                ("full population", "p3c__zmp"), ("full population, cut 2.0", "p3c__zmp2"),
                ("full population + regroup", "p3c__zmrp")]
    if any(name.startswith("p3c__") for name in runs):
        fig_variants(runs, p3c_rows, {42, 123, 7}, "phase3c_confirm.png", "Phase 3c · three seeds (mean ± sd)")
        fig_backdoor_variants(runs, p3c_rows, "phase3c_backdoor.png",
                              "Phase 3c · backdoor success, three seeds (bar = mean, dots = seeds)")
    fig_ceiling(runs)
    fig_p4(runs)
    fig_p4_paired(runs)
    fig_p4b(runs)
    fig_cascade(runs)
    fig_phase3_variants(runs)


if __name__ == "__main__":
    main()
