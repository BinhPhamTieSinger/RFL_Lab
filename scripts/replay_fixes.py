"""Offline evidence for the fixes queued on 2026-11-28 (report 8.2): the window rule and the two-signal test,
replayed on the union screen's logged scores before any live run.

  python scripts/replay_fixes.py outputs

Runs: GPU `gg*__*__r3wc_s<seed>` (union screen, benchmark settings) and Acer `x16_a05__*__r3wc_s<seed>` (union
screen under the timed backdoors of acer_c16). Two questions:
  1. who is "two-signal" (both signals' gated z below -1.5 in one of the last 5 rounds up to isolation): honest
     clients, backdoor attackers, or model-poisoning attackers? This decides what plus.rewind_agree and
     plus.sentence_single keep;
  2. how many attackers and honest clients each isolation rule catches: the screen now, 3 bad rounds in any 5,
     the low-side gate of 6 with the high side at 20, and both.
The replay's limits (scripts/replay_screen.py): the trajectory is the real run's, and a client the real run
isolated is not scored afterwards. The replay of the low-side gate predicted 3.2 honest isolations at 15 clients;
live it gave 5.3.
Writes docs/figures/replay_fixes.png and docs/results/replay_fixes.json."""
from __future__ import annotations

import glob
import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pr = _module("plot_results")
rs = _module("replay_screen")
plt = pr.plt
AGREE = 5
RULES = {"screen now (3 in a row, gate 8)": {},
         "3 bad rounds in any 5": {"window": 5},
         "gate 6, high side 20": {"low_gate": 6.0, "two_sided": False, "high_gate": 20.0},
         "both": {"window": 5, "low_gate": 6.0, "two_sided": False, "high_gate": 20.0}}
BACKDOOR = ("backdoor", "stealth", "sleeper", "onoff")
GPU_SETTINGS = {"gg15_a05": "D1", "gg20_a02": "D2", "gg30_a05": "30 cl., α 0.5", "gg50_a01": "50 cl., α 0.1"}


def load(outputs: Path) -> list[tuple[str, str, dict]]:
    runs = []
    for p in sorted(glob.glob(str(outputs / "gg*__*__r3wc_s*" / "metrics.json")) +
                    glob.glob(str(outputs / "x16_a05__*__r3wc_s*" / "metrics.json"))):
        if not re.search(r"__r3wc_s\d+$", Path(p).parent.name):
            continue
        m = json.loads(Path(p).read_text())
        if m.get("diagnostics") and "critical_scores" in m["diagnostics"][0]["score_stats"]:
            setting, cond = Path(p).parent.name.split("__")[:2]
            runs.append((setting, cond, m))
    return runs


def short(cell: str) -> str:
    """Axis label: the setting or the Acer condition, then the machine."""
    where, what = cell.split(": ")
    if where.startswith("Acer"):
        return {"onoff21": "on-off 2/1", "onoff3": "on-off 4/6", "sleeper3": "sleeper", "backdoor3": "backdoor",
                "clean": "15 clients"}[what] + "\nAcer"
    return where.removeprefix("GPU, ") + "\nGPU"


def size(setting: str) -> int:
    return 15 if setting.startswith("x16") else int(setting[2:4])


def two_signal(rows: list[dict], client: int, at: int) -> bool:
    for r in rows:
        if at - AGREE < r["round"] <= at:
            gw = -abs(r["zw"].get(client, 0.0)) if r["gw"] >= rs.GATE else 0.0
            gc = -abs(r["zc"].get(client, 0.0)) if r["gc"] >= rs.GATE else 0.0
            if gw < -rs.E_CUT and gc < -rs.E_CUT:
                return True
    return False


def main() -> None:
    pr.style()
    runs = load(Path(sys.argv[1] if len(sys.argv) > 1 else "outputs"))
    share: dict = defaultdict(lambda: defaultdict(lambda: [0, 0]))     # n -> group -> [two-signal, isolated]
    rule: dict = defaultdict(lambda: defaultdict(lambda: {"runs": 0, "honest": 0, "caught": 0, "attackers": 0}))
    for setting, cond, m in runs:
        bad, rows = set(m["attackers"]), rs.rounds(m)
        kind = "clean" if cond == "clean" else "backdoor" if cond.startswith(BACKDOOR) else "poison"
        for c, at in rs.replay(m, rows=rows).items():
            group = "honest" if c not in bad else f"{kind} attackers"
            share[size(setting)][group][0] += two_signal(rows, c, at)
            share[size(setting)][group][1] += 1
        if setting.startswith("x16"):
            cell = f"Acer, 15 clients: {cond}"
        elif setting in GPU_SETTINGS and kind != "poison":
            cell = f"GPU, {GPU_SETTINGS[setting]}: " + ("no attack" if kind == "clean" else "backdoor, stealth")
        else:
            continue
        for name, kw in RULES.items():
            iso = rs.replay(m, rows=rows, **kw)
            r = rule[cell][name]
            r["runs"] += 1
            r["honest"] += sum(c not in bad for c in iso)
            r["caught"] += sum(c in bad for c in iso)
            r["attackers"] += len(bad)
    out = {"two_signal_share": {n: {g: {"two_signal": v[0], "isolated": v[1]} for g, v in gs.items()}
                                for n, gs in sorted(share.items())},
           "rules": {c: dict(v) for c, v in sorted(rule.items())}}
    (ROOT / "docs" / "results" / "replay_fixes.json").write_text(json.dumps(out, indent=1))

    fig, axes = plt.subplots(1, 3, figsize=(19, 5), gridspec_kw={"width_ratios": [1, 1.4, 1.4]})
    ax = axes[0]
    groups = (("honest", pr.SLOT[0]), ("backdoor attackers", pr.SLOT[1]), ("poison attackers", pr.SLOT[2]))
    sizes = sorted(share)
    for j, (g, color) in enumerate(groups):
        xs = [i + (j - 1) * 0.27 for i in range(len(sizes))]
        vals = [100 * share[n][g][0] / share[n][g][1] if share[n][g][1] else 0.0 for n in sizes]
        ax.bar(xs, vals, 0.25, color=color, label=g.replace("poison", "model-poisoning"))
        for x, v, n in zip(xs, vals, sizes, strict=True):
            ax.text(x, v + 1, f"{v:.0f}\nof {share[n][g][1]}", ha="center", fontsize=6.5)
    ax.set_xticks(range(len(sizes)), [f"{n} clients" for n in sizes], fontsize=8)
    ax.set_ylim(0, 118)
    ax.set_yticks(range(0, 101, 20))
    ax.set_title("isolated clients that are two-signal (%):\n"
                 "both signals below −1.5 in one of their last 5 rounds", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="upper left", ncol=3, bbox_to_anchor=(0, 1.0), columnspacing=0.8)
    colors = [pr.INK, pr.SLOT[0], pr.SLOT[1], pr.SLOT[2]]
    for ax, field, cells, title in (
            (axes[1], "caught", [c for c in rule if "no attack" not in c and "clean" not in c],
             "attackers isolated (% of attackers); GPU: backdoor and stealth backdoor,\nAcer: 15 clients, the "
             "acer_c16 backdoors"),
            (axes[2], "honest", [c for c in rule if "no attack" in c or c.endswith("clean")],
             "honest clients isolated per run, no attack\n"
             "(D1 and the Acer: 15 clients, α 0.5; D2: 20 clients, α 0.2)")):
        w = 0.8 / len(RULES)
        for j, (name, color) in enumerate(zip(RULES, colors, strict=True)):
            for i, c in enumerate(cells):
                r = rule[c][name]
                v = 100 * r["caught"] / r["attackers"] if field == "caught" else r["honest"] / r["runs"]
                x = i + (j - 1.5) * w
                ax.bar(x, v, w * 0.95, color=color, label=name if i == 0 else None)
                ax.text(x, v + (1 if field == "caught" else 0.2), f"{v:.0f}" if field == "caught" else f"{v:.1f}",
                        ha="center", fontsize=6)
        ax.set_xticks(range(len(cells)), [short(c) for c in cells], fontsize=7)
        ax.set_title(title, fontsize=9.5)
        if field == "caught":
            ax.set_ylim(0, 118)
            ax.set_yticks(range(0, 101, 20))
            ax.legend(fontsize=7.5, loc="upper left", ncol=2)
    fig.suptitle("Offline replay on the union screen's logs: the window and the lower gate catch timed attackers "
                 "the screen misses; two-signal isolations separate attackers from honest clients at 20-30 "
                 "clients only",
                 fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    pr.save(fig, "replay_fixes.png")
    for c, v in sorted(rule.items()):
        print(c, {k: (x["honest"] / x["runs"], f"{x['caught']}/{x['attackers']}") for k, x in v.items()})
    for n, gs in sorted(share.items()):
        print(n, {g: f"{v[0]}/{v[1]}" for g, v in gs.items()})


if __name__ == "__main__":
    main()
