"""Offline replay of FedHEP-Guard's union screen on logged GPU runs (report 8.2): how well do its scores separate
attackers from honest clients, and what would other decision rules do?

  python scripts/replay_screen.py outputs

Every union-screen run (`*__r3wc_s*`) logs, per round, both signals' raw scores, their median and scale, and their
gap ratio. From these the script rebuilds the screen's robust z, gate and evidence; the replay of the real rule
reproduces the real isolations (checked on every run). Alternative rules are replayed on the same logs.

Two limits, both stated in the report: (1) a client the real run isolated is not scored afterwards, so a stricter
rule that has not isolated it by then is counted as sparing it (honest) or missing it (attacker); attacker catches
of stricter rules are therefore lower bounds. (2) The model trajectory is the real run's, not the one the
alternative rule would have produced.

Writes docs/figures/replay_separability.png, replay_rules.png and docs/results/replay_screen.json."""
from __future__ import annotations

import glob
import importlib.util
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("plot_results", Path(__file__).with_name("plot_results.py"))
pr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pr)
plt = pr.plt
Z_CUT, E_CUT, DECAY, PATIENCE, GATE = 2.5, 1.5, 0.7, 3, 8.0
COLORS = [pr.INK, pr.SLOT[1], pr.SLOT[2], pr.SLOT[3], pr.SLOT[0], "#7a3fb8", "#b8860b", "#2e8b57", "#c0392b"]


def load(outputs: Path) -> list[tuple[str, dict]]:
    runs = []
    for p in sorted(glob.glob(str(outputs / "gg*__r3wc_s*" / "metrics.json"))):
        m = json.loads(Path(p).read_text())
        if m.get("device") == "cuda" and m.get("diagnostics") and "critical_scores" in m["diagnostics"][0]["score_stats"]:
            runs.append((Path(p).parent.name, m))
    return runs


def kind(name: str) -> str:
    cond = name.split("__")[1]
    return "clean" if cond == "clean" else "backdoor" if cond.startswith(("backdoor", "stealth")) else "poison"


def rounds(m: dict) -> list[dict]:
    """Per round: each signal's robust z (rebuilt from the logged scores, median and scale) and gap ratio."""
    out = []
    for d in m["diagnostics"]:
        if not d["scores"]:
            continue
        ss = d["score_stats"]
        zw = {int(c): (v - ss["median"]) / ss["scale"] for c, v in d["scores"].items()}
        zc = {int(c): (v - ss["critical"]["median"]) / ss["critical"]["scale"]
              for c, v in ss["critical_scores"].items()}
        out.append({"round": d["round"], "zw": zw, "zc": zc, "gw": ss["gap_ratio"], "gc": ss["critical"]["gap_ratio"],
                    "iso": set(d["isolated"])})
    return out


def replay(m: dict, *, two_sided: bool = True, high_gate: float = GATE, cut: float = E_CUT,
           signals: tuple[str, ...] = ("w", "c"), low_gate: float = GATE, rows: list | None = None) -> dict[int, int]:
    """{client: isolation round} under a variant of the screen. The real rule: two_sided, high_gate = low_gate =
    GATE. two_sided False with high_gate G: the high side acts only when the gap ratio reaches G. low_gate L: the
    low side acts when the gap ratio reaches L (0: always, as in FedHEP+). rows: precomputed (or altered) rounds()."""
    hi = GATE if two_sided and high_gate <= GATE else high_gate
    ev: dict[int, float] = {}
    streak: dict[int, int] = {}
    iso: dict[int, int] = {}
    for r in rows if rows is not None else rounds(m):
        sig = {"w": (r["zw"], r["gw"]), "c": (r["zc"], r["gc"])}
        for c in r["zw"]:
            if c in iso:
                continue
            v = 0.0
            for s in signals:
                z, g = sig[s]
                x = z.get(c, 0.0)
                if x < 0 and g >= low_gate:
                    v = min(v, x)
                elif x > 0 and g >= hi:
                    v = min(v, -x)
            ev[c] = DECAY * ev.get(c, 0.0) + (1 - DECAY) * v
            streak[c] = streak.get(c, 0) + 1 if v < -cut else 0
            if ev[c] < -cut and streak[c] >= PATIENCE:
                iso[c] = r["round"]
    return iso


RULES = {"FedHEP-Guard's screen (now)": {},
         "high side only if gap ratio ≥ 20": {"two_sided": False, "high_gate": 20.0},
         "high side only if gap ratio ≥ 30": {"two_sided": False, "high_gate": 30.0},
         "high side ≥ 30, cut 2.0": {"two_sided": False, "high_gate": 30.0, "cut": 2.0},
         "high side ≥ 30, whole-update signal only": {"two_sided": False, "high_gate": 30.0, "signals": ("w",)},
         "two-sided, cut 2.0": {"cut": 2.0},
         "low side ungated, high side ≥ 8": {"low_gate": 0.0},
         "low side ungated, high side ≥ 20": {"low_gate": 0.0, "two_sided": False, "high_gate": 20.0},
         "low side if gap ratio ≥ 6, high side ≥ 20": {"low_gate": 6.0, "two_sided": False, "high_gate": 20.0}}


def auroc(bad: list[float], good: list[float]) -> float:
    b, g = np.asarray(bad), np.asarray(good)
    return float((b[:, None] > g[None, :]).mean() + 0.5 * (b[:, None] == g[None, :]).mean())


def separability(runs) -> dict:
    """Mean per-round AUROC (rounds 1-10, attackers still scored) of backdoor attackers against honest clients."""
    feats = {"two-sided |z|, whole update": lambda w, c, ew, ec: abs(w),
             "low side −z, whole update": lambda w, c, ew, ec: -w,
             "two-sided |z|, critical layers": lambda w, c, ew, ec: abs(c),
             "low side −z, critical layers": lambda w, c, ew, ec: -c,
             "low side, smoothed over rounds (EMA 0.7), whole update": lambda w, c, ew, ec: -ew}
    acc: dict = defaultdict(lambda: defaultdict(list))
    for name, m in runs:
        if kind(name) != "backdoor":
            continue
        n = int(name[2:4])
        bad = set(m["attackers"])
        ew: dict[int, float] = {}
        ec: dict[int, float] = {}
        for r in rounds(m):
            for c, v in r["zw"].items():
                ew[c] = DECAY * ew.get(c, v) + (1 - DECAY) * v
                ec[c] = DECAY * ec.get(c, r["zc"].get(c, 0.0)) + (1 - DECAY) * r["zc"].get(c, 0.0)
            att = [c for c in r["zw"] if c in bad]
            if not att or r["round"] > 10:
                continue
            for f, fn in feats.items():
                val = {c: fn(r["zw"][c], r["zc"].get(c, 0.0), ew[c], ec[c]) for c in r["zw"]}
                acc[n][f].append(auroc([val[c] for c in att], [val[c] for c in r["zw"] if c not in bad]))
    return {n: {f: statistics.fmean(v) for f, v in fs.items()} for n, fs in acc.items()}


def evaluate(runs) -> dict:
    res: dict = defaultdict(lambda: defaultdict(lambda: {"runs": 0, "honest": 0, "caught": 0, "attackers": 0}))
    for name, m in runs:
        n, k, bad = int(name[2:4]), kind(name), set(m["attackers"])
        for rule, kw in RULES.items():
            iso = replay(m, **kw)
            cell = res[rule][f"{n}/{k}"]
            cell["runs"] += 1
            cell["honest"] += sum(c not in bad for c in iso)
            cell["caught"] += sum(c in bad for c in iso)
            cell["attackers"] += len(bad)
    return res


def fig_separability(sep: dict) -> None:
    ns = sorted(sep)
    feats = list(next(iter(sep.values())))
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    w = 0.8 / len(feats)
    for j, f in enumerate(feats):
        vals = [sep[n][f] for n in ns]
        xs = [i + (j - len(feats) / 2 + 0.5) * w for i in range(len(ns))]
        ax.bar(xs, vals, w * 0.95, color=COLORS[j % len(COLORS)], label=f)
        for x, v in zip(xs, vals, strict=True):
            ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=6.5)
    ax.axhline(0.5, color=pr.INK2, linestyle=":", linewidth=0.8)
    ax.set_xticks(range(len(ns)), [f"{n} clients" for n in ns])
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("AUROC, backdoor attackers vs honest")
    ax.set_title("How well the screen's scores separate backdoor attackers from honest clients, per round "
                 "(GPU grid, rounds 1–10)", fontsize=10)
    ax.legend(fontsize=7.5, ncol=2, loc="lower left")
    fig.tight_layout()
    pr.save(fig, "replay_separability.png")


def fig_rules(res: dict) -> None:
    ns = sorted({int(k.split("/")[0]) for cells in res.values() for k in cells if k.endswith("backdoor")})
    fig, axes = plt.subplots(1, len(ns), figsize=(4.6 * len(ns), 4.2), squeeze=False)
    for ax, n in zip(axes[0], ns, strict=True):
        for j, (rule, cells) in enumerate(res.items()):
            clean, bd = cells.get(f"{n}/clean"), cells.get(f"{n}/backdoor")
            if not clean or not bd:
                continue
            x, y = clean["honest"] / clean["runs"], 100 * bd["caught"] / bd["attackers"]
            ax.scatter(x, y, s=90 if j == 0 else 40, color=COLORS[j % len(COLORS)], marker="*" if j == 0 else "o",
                       label=rule, zorder=3)
        ax.set_xlabel("honest clients isolated per run, no attack")
        ax.set_ylabel("backdoor attackers caught (%, lower bound)")
        ax.set_title(f"{n} clients (GPU, every α run)", fontsize=10)
        ax.set_ylim(0, 100)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, ncol=5, loc="lower center", frameon=False)
    fig.suptitle("Replayed decision rules: honest clients blamed (no attack) against backdoor attackers caught",
                 fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    pr.save(fig, "replay_rules.png")


def main() -> None:
    pr.style()
    runs = load(Path(sys.argv[1] if len(sys.argv) > 1 else "outputs"))
    real = sum(replay(m) == {c: next(d["round"] for d in m["diagnostics"] if c in d["isolated"])
                             for c in {c for d in m["diagnostics"] for c in d["isolated"]}} for _, m in runs)
    sep, res = separability(runs), evaluate(runs)
    out = {"runs": len(runs), "replay_matches_real": real, "separability": sep,
           "rules": {r: dict(c) for r, c in res.items()}}
    (ROOT / "docs" / "results" / "replay_screen.json").write_text(json.dumps(out, indent=1))
    fig_separability(sep)
    fig_rules(res)
    print(f"{len(runs)} runs; replay reproduces the real isolations in {real}")
    for n, fs in sorted(sep.items()):
        print(n, {f: round(v, 3) for f, v in fs.items()})
    for rule, cells in res.items():
        print(f"{rule:44s}", " | ".join(f"{k} h{c['honest'] / c['runs']:.1f}"
                                         + (f" a{100 * c['caught'] / c['attackers']:.0f}%" if c["attackers"] else "")
                                         for k, c in sorted(cells.items())))


if __name__ == "__main__":
    main()
