"""Phase 7 (Acer, acer_c10_peer.yaml): the peer reference against the global median.

Two readings:
  * paired (same training run): every run logs both references' scores, one acted on and one in the shadow, so
    their z-scores are compared round by round on identical updates. Honest outliers = share of honest
    client-rounds with |z| > 2.5; separation = per-round AUROC of attackers against honest clients on -|z|
    (the screen is two-sided), averaged over the rounds in which attackers are still scored;
  * acting: accuracy, backdoor success and honest clients isolated per run, median arm against peer arm.
Writes docs/figures/phase7_peer_paired.png, phase7_peer_outcomes.png and docs/results/phase7_peer.json.

Usage: python scripts/plot_phase7.py outputs
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

_spec = importlib.util.spec_from_file_location("plot_results", Path(__file__).with_name("plot_results.py"))
assert _spec and _spec.loader
pr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pr)
plt = pr.plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rfl.defense.screening import robust_z  # noqa: E402

SEEDS = (42, 123, 7)
ALPHAS = (("a01", "α = 0.1"), ("a05", "α = 0.5"))
ARMS = (("r3wc_shp", "median", "peer"), ("r3wc_peer", "peer", "median"))   # (arm, acts on, shadow)
SIGNALS = ("weights", "critical")


def _z(scores: list[float]) -> list[float]:
    return robust_z(torch.tensor(scores, dtype=torch.float64)).tolist()


def _auroc(bad: list[float], good: list[float]) -> float:
    """P(attacker ranks worse than an honest client), ties count half; inputs are 'badness' values."""
    wins = sum((b > g) + 0.5 * (b == g) for b in bad for g in good)
    return wins / (len(bad) * len(good))


def paired(m: dict, acts: str) -> dict[str, dict[str, dict[str, float]]]:
    """{signal: {reference: {"honest_outliers", "auroc"}}} for one run."""
    bad = set(m["attackers"])
    out: dict[str, dict[str, dict[str, float]]] = {}
    for sig in SIGNALS:
        acc: dict[str, dict[str, list[float]]] = {r: {"flag": [], "auc": []} for r in ("median", "peer")}
        for d in m["diagnostics"]:
            stats = d["score_stats"] if sig == "weights" else d["score_stats"].get("critical")
            if not stats or "shadow_scores" not in stats:
                continue
            clients = [int(c) for c in d["scores"]]
            primary = list(d["scores"].values()) if sig == "weights" else d["score_stats"]["critical_scores"]
            primary = list(primary.values()) if isinstance(primary, dict) else primary
            for ref, scores in ((acts, primary), ({"median": "peer", "peer": "median"}[acts],
                                                  stats["shadow_scores"])):
                z = dict(zip(clients, _z(scores), strict=True))
                honest = [abs(v) for c, v in z.items() if c not in bad]
                acc[ref]["flag"] += [float(v > 2.5) for v in honest]
                attackers = [abs(v) for c, v in z.items() if c in bad]
                if attackers and honest:
                    acc[ref]["auc"].append(_auroc(attackers, honest))
        out[sig] = {r: {"honest_outliers": sum(v["flag"]) / max(1, len(v["flag"])),
                        "auroc": sum(v["auc"]) / len(v["auc"]) if v["auc"] else float("nan")}
                    for r, v in acc.items()}
    return out


def outcome(m: dict) -> dict[str, float]:
    last = m["diagnostics"][-1]
    bad = set(m["attackers"])
    return {"accuracy": m["final"]["pooled"], "asr": m["history"][-1].get("asr") or 0.0,
            "honest_isolated": len(set(last["isolated"]) - bad), "attackers_isolated": len(set(last["isolated"]) & bad)}


def _mean(v: list[float]) -> float:
    v = [x for x in v if x == x]
    return sum(v) / len(v) if v else float("nan")


def summarize(runs: dict) -> dict:
    res: dict = {}
    for a, _ in ALPHAS:
        for attack in ("clean", "backdoor3"):
            for arm, acts, _ in ARMS:
                ms = [m for m in runs.get(f"x10_{a}__{attack}__{arm}", []) if m["seed"] in SEEDS]
                if not ms:
                    continue
                key = f"{a}/{attack}/{arm}"
                ps = [paired(m, acts) for m in ms]
                res[key] = {"seeds": [m["seed"] for m in ms],
                            "paired": {sig: {ref: {k: _mean([p[sig][ref][k] for p in ps])
                                                   for k in ("honest_outliers", "auroc")}
                                             for ref in ("median", "peer")} for sig in SIGNALS},
                            "outcome": {k: _mean([outcome(m)[k] for m in ms]) for k in outcome(ms[0])},
                            "per_seed": [outcome(m) for m in ms]}
    return res


def fig_paired(res: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    width = 0.2
    groups = [(a, attack) for a, _ in ALPHAS for attack in ("clean", "backdoor3")]
    labels = [f"{dict(ALPHAS)[a]}\n{'no attack' if t == 'clean' else 'backdoor, 3'}" for a, t in groups]
    panels = [("honest_outliers", "clean+backdoor", "honest client-rounds with |z| > 2.5", axes[0]),
              ("auroc", "backdoor", "attacker separation (AUROC on |z|)", axes[1])]
    for metric, _, title, ax in panels:
        bars = [(sig, ref) for sig in SIGNALS for ref in ("median", "peer")]
        for j, (sig, ref) in enumerate(bars):
            vals = []
            for a, attack in groups:
                rows = [res[k]["paired"][sig][ref][metric] for k in res if k.startswith(f"{a}/{attack}/")]
                vals.append(_mean(rows) if not (metric == "auroc" and attack == "clean") else float("nan"))
            color = pr.SLOT[0] if ref == "median" else pr.SLOT[1]
            ax.bar([i + (j - 1.5) * width for i in range(len(groups))], [0 if v != v else v for v in vals],
                   width - 0.02, color=color, alpha=1.0 if sig == "weights" else 0.55,
                   label=f"{sig.replace('weights', 'whole update').replace('critical', 'critical layers')}, "
                         f"{ref} reference")
        ax.set_xticks(range(len(groups)), labels, fontsize=8)
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("share")
    axes[1].set_ylim(0.4, 1.0)
    axes[1].axhline(0.5, color="#9a9893", linewidth=0.8, linestyle=":")
    axes[0].legend(fontsize=7.5, loc="upper left")
    ax = axes[2]
    for j, (arm, acts, _) in enumerate(ARMS):
        vals = [res.get(f"{a}/clean/{arm}", {}).get("outcome", {}).get("honest_isolated", float("nan"))
                for a, _ in ALPHAS]
        ax.bar([i + (j - 0.5) * 0.35 for i in range(2)], [0 if v != v else v for v in vals], 0.33,
               color=pr.SLOT[0] if acts == "median" else pr.SLOT[1], label=f"acting on the {acts} reference")
    ax.set_xticks(range(2), [label for _, label in ALPHAS])
    ax.set_title("honest clients isolated per run, no attack", fontsize=10)
    ax.legend(fontsize=7.5)
    fig.suptitle("Phase 7 (Acer): peer reference against the global median; left two panels compare both "
                 "references on the same runs", fontsize=10.5)
    fig.tight_layout()
    pr.save(fig, "phase7_peer_paired.png")


def fig_outcomes(res: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    metrics = [("accuracy", "accuracy (%)", "clean"), ("accuracy", "accuracy under backdoor (%)", "backdoor3"),
               ("asr", "backdoor success (%)", "backdoor3")]
    for ax, (metric, title, attack) in zip(axes, metrics, strict=True):
        for j, (arm, acts, _) in enumerate(ARMS):
            per = [res.get(f"{a}/{attack}/{arm}", {}).get("per_seed", []) for a, _ in ALPHAS]
            mu = [_mean([p[metric] for p in ps]) for ps in per]
            xs = [i + (j - 0.5) * 0.35 for i in range(2)]
            ax.bar(xs, [0 if v != v else v for v in mu], 0.33, color=pr.SLOT[0] if acts == "median" else pr.SLOT[1],
                   label=f"acting on the {acts} reference")
            for x, ps in zip(xs, per, strict=True):
                ax.scatter([x] * len(ps), [p[metric] for p in ps], color=pr.INK, s=10, zorder=3)
        ax.set_xticks(range(2), [label for _, label in ALPHAS])
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=7.5)
    fig.suptitle("Phase 7 (Acer): outcomes when the screen acts on each reference (3 seeds, dots = seeds)",
                 fontsize=10.5)
    fig.tight_layout()
    pr.save(fig, "phase7_peer_outcomes.png")


def main() -> None:
    pr.style()
    runs = pr.load(Path(sys.argv[1]))
    res = summarize(runs)
    if not res:
        return print("no x10 runs")
    out = Path(__file__).resolve().parents[1] / "docs" / "results" / "phase7_peer.json"
    out.write_text(json.dumps(res, indent=1))
    fig_paired(res)
    fig_outcomes(res)
    for k, v in res.items():
        p = v["paired"]["weights"]
        print(k, "seeds", v["seeds"], {kk: round(vv, 2) for kk, vv in v["outcome"].items()},
              "| honest |z|>2.5 med/peer", round(p["median"]["honest_outliers"], 3), round(p["peer"]["honest_outliers"], 3),
              "| AUROC med/peer", round(p["median"]["auroc"], 3), round(p["peer"]["auroc"], 3))


if __name__ == "__main__":
    main()
