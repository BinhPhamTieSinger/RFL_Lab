"""The benchmark (report 6.1): every model and variant, the same eight conditions, every setting, one machine.

  python scripts/plot_benchmark.py outputs

The model list, settings and conditions come from scripts/make_benchmark.py. A GPU-server run fills a benchmark
cell when its configuration is the cell's (settings that a model does not read are ignored), whatever its name,
so runs from the large grid and earlier GPU studies count. Runs from other machines never do. Cells with no runs
stay empty in every chart.

Writes, under docs/figures/bench/:
  overview_<setting>.png     every model x every condition: accuracy, backdoor success, detection F1, honest
                             clients isolated
  <setting>__<condition>.png one condition in detail, once at least one model has run it: accuracy, attackers
                             caught, detection F1, honest clients isolated, backdoor success, and in which round
                             attackers and honest clients are isolated
docs/results/benchmark.json with every number drawn, and the chart index of report 6.1 (between the markers
"[//]: # (BENCH:index START)" and "[//]: # (BENCH:index END)" in docs/report/06_results.md; link-reference
lines like these render as nothing in any Markdown viewer)."""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rfl.experiment.config import load_config  # noqa: E402
from rfl.experiment.sweep import expand  # noqa: E402


def _module(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pr = _module("plot_results")
mb = _module("make_benchmark")
plt = pr.plt
FIG = ROOT / "docs" / "figures" / "bench"
SEEDS = (42, 123, 7)
ROUNDS = 30
CONDITIONS = [("clean", "No attack"), ("sign", "Sign flip"), ("label", "Label flip"), ("noise", "Random noise"),
              ("alie", "ALIE"), ("minmax", "Min-Max"), ("backdoor", "Backdoor"), ("stealth", "Stealth backdoor")]
BACKDOORS = ("backdoor", "stealth")
FAMILIES = [("Baselines", "#9a9893"), ("Published FedHEP models", "#52514e"), ("FedHEP+ and its variants", pr.SLOT[1]),
            ("Established robust rules", pr.SLOT[0]), ("FedHEP-Guard ladder and variants", pr.SLOT[2]),
            ("FedHEP-Guard", pr.SLOT[3])]
MODELS = list(mb.MODELS)
LABEL = {k: v[1] for k, v in mb.MODELS.items()}
FAMILY = {k: v[0] for k, v in mb.MODELS.items()}
NO_SERVER = set(mb.FEDAVG_FAMILY)      # no ALIE, Min-Max or stealth backdoor
EVAL_ONLY = {("fedhep", "exit_guard"), ("fedhep", "head_agreement"), ("run", "output_dir"), ("run", "device"),
             ("run", "seed"), ("run", "eval_every"), ("data", "root")}


def setting_tag(n: int, alpha: float) -> str:
    return mb.tag(n, alpha)


def cell_key(cfg: dict) -> str:
    """The configuration a run's results depend on: sections the method does not read and evaluation-only
    switches are dropped, so the same experiment matches under any run name."""
    c = {s: (dict(v) if isinstance(v, dict) else v) for s, v in cfg.items() if s != "name"}
    for s, k in EVAL_ONLY:
        c.get(s, {}).pop(k, None)
    method = c["method"]
    if method != "fedhep_plus":
        c.pop("plus", None)
    elif c["plus"].get("global_rule") != "arc_nnm_tm":
        c["plus"].pop("global_f", None)
        c["plus"].pop("global_f_fixed", None)
    if method != "fedhep_robust":
        c.pop("robust", None)
    if method not in ("fedavg", "multikrum"):
        c.pop("baseline", None)
    a = c["attack"]
    if a.get("kind") is None:
        c["attack"] = None
    else:
        if a["kind"] != "alie":
            a.pop("alie_z", None)
        if a["kind"] not in ("alie", "min_max"):
            a.pop("knowledge", None)
        if a["kind"] not in ("backdoor", "stealth_backdoor"):
            a.pop("backdoor_target", None)
            a.pop("backdoor_fraction", None)
        a.pop("rate", None)
    return json.dumps(c, sort_keys=True, default=str)


def cells() -> dict[str, tuple[str, str, str]]:
    """cell key -> (setting tag, condition, model), from the benchmark sweep files."""
    out = {}
    for f in sorted((ROOT / "configs" / "sweeps").glob("bench_*.yaml")):
        for name, cfg in expand(yaml.safe_load(f.read_text())):
            setting, cond, model = name.split("__")
            out[cell_key(dataclasses.asdict(cfg))] = (setting, cond.rstrip("0123456789"), model)
    return out


def summarize(m: dict) -> dict:
    """One run's numbers: accuracy, backdoor success, detection F1 (mean over rounds of the F1 of the clients left
    out that round), honest clients excluded per round, and the cumulative isolations per round."""
    bad = set(m["attackers"])
    det = m.get("detection") or []
    f1 = [d["f1"] for d in det if d.get("f1") is not None]
    iso_rounds = [set(d.get("isolated") or []) for d in det]
    isolating = any("isolated" in d for d in det)
    last = iso_rounds[-1] if iso_rounds else set()
    h = m["history"][-1]
    return {"seed": m["seed"], "acc": m["final"]["pooled"], "asr": h.get("asr"),
            "asr_guard": (h.get("guarded") or {}).get("asr"),
            "f1": statistics.fmean(f1) if bad and f1 else None,
            "excluded": statistics.fmean(d.get("false_exclusion_rate") or 0.0 for d in det) if det else None,
            "isolating": isolating, "attackers": len(bad),
            "caught": len(last & bad) if isolating else None, "honest": len(last - bad) if isolating else 0,
            "caught_by_round": [len(s & bad) for s in iso_rounds], "honest_by_round": [len(s - bad) for s in iso_rounds]}


def collect(outputs: Path) -> dict:
    lookup = cells()
    runs: dict = defaultdict(dict)    # (setting, cond, model) -> {seed: summary}
    for path in sorted(outputs.glob("*/metrics.json")):
        try:
            m = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if m.get("device") != "cuda" or m.get("seed") not in SEEDS or not m.get("final"):
            continue
        cfg_path = path.with_name("config.yaml")
        if not cfg_path.exists():
            continue
        try:
            key = cell_key(dataclasses.asdict(load_config(cfg_path)))
        except (TypeError, ValueError, KeyError):
            continue
        cell = lookup.get(key)
        if cell and m["seed"] not in runs[cell]:
            runs[cell][m["seed"]] = summarize(m)
    return runs


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else None


def stats(rs: list[dict]) -> dict:
    """Mean over seeds; rounds curves averaged; attackers caught as a share of all attackers in the runs."""
    acc = [r["acc"] for r in rs]
    caught = [r["caught"] for r in rs if r["caught"] is not None]
    n_att = sum(r["attackers"] for r in rs if r["caught"] is not None)
    rounds = min(len(r["caught_by_round"]) for r in rs)
    return {"seeds": len(rs), "acc": statistics.fmean(acc), "acc_sd": statistics.stdev(acc) if len(acc) > 1 else 0.0,
            "asr": _mean([r["asr"] for r in rs]), "asr_seeds": [r["asr"] for r in rs],
            "asr_guard": _mean([r["asr_guard"] for r in rs]), "f1": _mean([r["f1"] for r in rs]),
            "excluded": _mean([r["excluded"] for r in rs]), "isolating": any(r["isolating"] for r in rs),
            "caught": (sum(caught) / n_att) if n_att else None, "honest": statistics.fmean(r["honest"] for r in rs),
            "caught_by_round": [statistics.fmean(r["caught_by_round"][i] / max(1, r["attackers"]) for r in rs)
                                for i in range(rounds)],
            "honest_by_round": [statistics.fmean(r["honest_by_round"][i] for r in rs) for i in range(rounds)]}


def table(runs: dict) -> dict:
    out: dict = {}
    for (setting, cond, model), by_seed in runs.items():
        out.setdefault(setting, {}).setdefault(cond, {})[model] = stats([by_seed[s] for s in SEEDS if s in by_seed])
    return out


def applies(model: str, cond: str) -> bool:
    return not (model in NO_SERVER and cond in ("alie", "minmax", "stealth"))


def _family_ticks(ax, models: list[str]) -> None:
    ax.set_yticks(range(len(models)), [LABEL[k] for k in models], fontsize=7.5)
    for tick, k in zip(ax.get_yticklabels(), models, strict=True):
        tick.set_color(FAMILIES[FAMILY[k]][1] if FAMILY[k] else pr.INK)
        if FAMILY[k] == 5:
            tick.set_fontweight("bold")


def _separators(axes, models: list[str]) -> None:
    """A line between model families, across every panel."""
    for ax in axes:
        for i in range(1, len(models)):
            if FAMILY[models[i]] != FAMILY[models[i - 1]]:
                ax.axhline(i - 0.5, color=pr.INK2, linewidth=0.6)


def _heat(ax, grid, cmap, vmin, vmax, fmt, models, cols, empty):
    import numpy as np
    ax.grid(False)   # grid lines would cross the numbers in the cells
    arr = np.array([[np.nan if v is None else v for v in row] for row in grid], dtype=float)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#efeeeb")
    ax.imshow(arr, aspect="auto", cmap=cm, vmin=vmin, vmax=vmax)
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            if v is None:
                ax.text(j, i, empty[i][j], ha="center", va="center", fontsize=5.5, color="#b5b3ae")
            else:
                shade = (v - vmin) / (vmax - vmin) if vmax > vmin else 0
                ax.text(j, i, fmt(v), ha="center", va="center", fontsize=6,
                        color="white" if (shade > 0.6 if cmap != "RdYlGn" else abs(shade - 0.5) > 0.35) else pr.INK)
    ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(models)), [])


def fig_overview(tab: dict, setting: str, label: str) -> None:
    models = MODELS
    data = tab.get(setting, {})
    get = lambda c, k, f: (data.get(c, {}).get(k) or {}).get(f)  # noqa: E731
    empty = lambda conds: [["n/a" if not applies(k, c) else "–" for c in conds] for k in models]  # noqa: E731
    conds = [c for c, _ in CONDITIONS]
    titles = [t for _, t in CONDITIONS]
    attacks = conds[1:]
    panels = [("accuracy (%)", conds, titles, "acc", "RdYlGn", None, None, lambda v: f"{v:.1f}"),
              ("backdoor\nsuccess (%)", list(BACKDOORS), ["Backdoor", "Stealth"], "asr", "Reds", 0, 100,
               lambda v: f"{v:.0f}"),
              ("detection F1\n(mean over rounds)", attacks, titles[1:], "f1", "Blues", 0, 1, lambda v: f"{v:.2f}"),
              ("honest clients\nisolated per run", conds, titles, "honest", "Oranges", 0, None, lambda v: f"{v:.1f}")]
    widths = [len(p[1]) for p in panels]
    fig, axes = plt.subplots(1, 4, figsize=(4.2 + 0.52 * sum(widths), 1.6 + 0.2 * len(models)),
                             gridspec_kw={"width_ratios": widths}, sharey=True)
    for ax, (title, cs, ts, field, cmap, vmin, vmax, fmt) in zip(axes, panels, strict=True):
        grid = [[get(c, k, field) if applies(k, c) else None for c in cs] for k in models]
        vals = [v for row in grid for v in row if v is not None]
        lo = vmin if vmin is not None else (min(vals) if vals else 0)
        hi = vmax if vmax is not None else (max(vals) if vals else 1)
        _heat(ax, grid, cmap, lo, max(hi, lo + 1e-6), fmt, models, ts, empty(cs))
        ax.set_title(title, fontsize=9.5)
    _family_ticks(axes[0], models)
    _separators(axes, models)
    seeds = sorted({v["seeds"] for c in data.values() for v in c.values()})
    filled = sum(1 for c in data.values() for _ in c)
    fig.suptitle(f"Benchmark, {label}: every model under every condition (GPU server, mean of 3 seeds; "
                 f"{filled} of {sum(applies(k, c) for k in models for c in conds)} cells run"
                 + (f"; some cells have fewer seeds: {seeds}" if seeds and seeds != [3] else "") + ")",
                 fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"overview_{setting}.png", dpi=120)
    plt.close(fig)


def fig_detail(tab: dict, setting: str, label: str, cond: str, title: str) -> None:
    import numpy as np
    models = [k for k in MODELS if applies(k, cond)]
    data = tab.get(setting, {}).get(cond, {})
    backdoor = cond in BACKDOORS
    attacked = cond != "clean"
    cols = ["acc"] + (["asr"] if backdoor else []) + (["caught"] if attacked else []) + ["honest", "hr"] \
        + (["cr"] if attacked else [])
    ratios = {"acc": 1.2, "asr": 1.0, "caught": 1.0, "honest": 0.9, "hr": 1.6, "cr": 1.6}
    fig, axes = plt.subplots(1, len(cols), figsize=(3.2 + 2.6 * sum(ratios[c] for c in cols), 1.4 + 0.2 * len(models)),
                             gridspec_kw={"width_ratios": [ratios[c] for c in cols]}, sharey=True)
    ax_of = dict(zip(cols, axes, strict=True))
    ys = range(len(models))
    color = [FAMILIES[FAMILY[k]][1] for k in models]

    def bars(ax, vals, fmt, xlabel, xmax=None, errs=None, missing=None):
        for y, v in enumerate(vals):
            if v is None:
                ax.text(0.01, y, missing[y] if missing else "not run", transform=ax.get_yaxis_transform(),
                        va="center", fontsize=6, color="#b5b3ae")
                continue
            ax.barh(y, v, color=color[y], height=0.72, xerr=(errs[y] if errs else None), capsize=1.5,
                    error_kw={"elinewidth": 0.6, "ecolor": pr.INK2})
            ax.text(v + (errs[y] if errs and errs[y] else 0), y, " " + fmt(v), va="center", fontsize=6)
        ax.set_xlabel(xlabel, fontsize=8)
        top = max([v for v in vals if v is not None] or [1])
        ax.set_xlim(0, xmax or top * 1.25 + 1e-6)
        ax.grid(axis="y", visible=False)

    get = lambda k, f: (data.get(k) or {}).get(f)  # noqa: E731
    bars(ax_of["acc"], [get(k, "acc") for k in models], lambda v: f"{v:.1f}", "accuracy of honest clients (%)",
         errs=[get(k, "acc_sd") for k in models])
    if backdoor:
        ax = ax_of["asr"]
        bars(ax, [get(k, "asr") for k in models], lambda v: f"{v:.1f}", "backdoor success (%), dots = seeds", 100)
        for y, k in enumerate(models):
            seeds = get(k, "asr_seeds") or []
            ax.scatter([s for s in seeds if s is not None], [y] * len([s for s in seeds if s is not None]), s=5,
                       color=pr.INK, zorder=3)
    if attacked:
        ax = ax_of["caught"]
        vals = [None if get(k, "acc") is None else (get(k, "caught") if get(k, "isolating") else None)
                for k in models]
        bars(ax, [None if v is None else 100 * v for v in vals], lambda v: f"{v:.0f}%",
             "attackers isolated by round 30 (%)\n◆ detection F1 (top axis)", 100,
             missing=["not run" if get(k, "acc") is None else "never isolates" for k in models])
        twin = ax.twiny()
        f1 = [get(k, "f1") for k in models]
        twin.scatter([v for v in f1 if v is not None], [y for y, v in enumerate(f1) if v is not None], marker="D",
                     s=10, color=pr.INK, zorder=4)
        twin.set_xlim(0, 1)
        twin.tick_params(labelsize=7)
    bars(ax_of["honest"], [None if get(k, "acc") is None else get(k, "honest") for k in models],
         lambda v: f"{v:.1f}", "honest clients isolated per run")

    def heat(ax, field, title, vmax, cmap):
        arr = np.full((len(models), ROUNDS), np.nan)
        for y, k in enumerate(models):
            row = get(k, field)
            if row:
                arr[y, :len(row)] = row[:ROUNDS]
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad("#efeeeb")
        im = ax.imshow(arr, aspect="auto", cmap=cm, vmin=0, vmax=vmax, extent=(0.5, ROUNDS + 0.5, len(models) - 0.5,
                                                                                -0.5))
        ax.set_xlabel(title, fontsize=8)
        ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01).ax.tick_params(labelsize=6)

    vmax = max([max(get(k, "honest_by_round") or [0]) for k in models] + [1])
    heat(ax_of["hr"], "honest_by_round", "round  ·  honest clients isolated so far (mean of seeds)", vmax, "Oranges")
    if attacked:
        heat(ax_of["cr"], "caught_by_round", "round  ·  share of attackers isolated so far", 1, "Blues")
    _family_ticks(axes[0], models)
    _separators(axes, models)
    filled = sum(1 for k in models if get(k, "acc") is not None)
    fig.suptitle(f"{title}, {label} (GPU server, mean of 3 seeds; {filled} of {len(models)} models run)",
                 fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(FIG / f"{setting}__{cond}.png", dpi=95)
    plt.close(fig)


CHAPTER = ROOT / "docs" / "report" / "06_results.md"
EMBED = ("gg15_a05", "gg20_a02")   # the report's two settings: detail charts shown, the others linked


def report_block(tab: dict) -> str:
    lines = []
    for n, a, label in mb.SETTINGS:
        setting = setting_tag(n, a)
        data = tab.get(setting, {})
        total = sum(applies(k, c) for k in MODELS for c, _ in CONDITIONS)
        filled = sum(len(v) for v in data.values())
        name = (f"{label}: " if label in ("D1", "D2") else "") + f"{n} clients, α = {a:g}, {round(0.2 * n)} attackers" \
            + (" (the paper's setting)" if a == 0.05 else "")
        lines += [f"#### {name} (`{setting}`)", "",
                  f"{filled} of {total} cells run.", "",
                  f"![Benchmark overview, {label}](../figures/bench/overview_{setting}.png)", ""]
        present = [(c, t) for c, t in CONDITIONS if (FIG / f"{setting}__{c}.png").exists()]
        if not present:
            lines += ["Detail charts: none yet.", ""]
        elif setting in EMBED:
            lines += [f"![{t}, {label}](../figures/bench/{setting}__{c}.png)\n" for c, t in present]
        else:
            lines += ["Detail charts: " + " · ".join(f"[{t}](../figures/bench/{setting}__{c}.png)"
                                                     for c, t in present), ""]
    return "\n".join(lines)


def write_report(tab: dict) -> None:
    start, end = "[//]: # (BENCH:index START)", "[//]: # (BENCH:index END)"
    text = CHAPTER.read_text() if CHAPTER.exists() else ""
    if start not in text or end not in text:
        return
    before, rest = text.split(start, 1)
    after = rest.split(end, 1)[1]
    CHAPTER.write_text(f"{before}{start}\n{report_block(tab)}\n{end}{after}")
    print(f"updated {CHAPTER.relative_to(ROOT)}")


def main() -> None:
    pr.style()
    tab = table(collect(Path(sys.argv[1] if len(sys.argv) > 1 else "outputs")))
    labels = {setting_tag(n, a): lab for n, a, lab in mb.SETTINGS}
    for setting, label in labels.items():
        fig_overview(tab, setting, label)
        for cond, title in CONDITIONS:
            if tab.get(setting, {}).get(cond):   # a detail chart once at least one model has run
                fig_detail(tab, setting, label, cond, title)
    out = ROOT / "docs" / "results" / "benchmark.json"
    out.write_text(json.dumps(tab, indent=1))
    write_report(tab)
    for setting, label in labels.items():
        n = sum(len(v) for v in tab.get(setting, {}).values())
        print(f"{setting} ({label}): {n} cells with runs")


if __name__ == "__main__":
    main()
