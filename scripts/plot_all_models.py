"""Every model ever run in one setting on one machine, one chart per attack (report 7.1).

A run belongs to the setting when it has 15 clients, alpha 0.5, 10,000 training images and 30 rounds, and, under
attack, exactly 3 attackers placed at random with no boost and no stop. Models are told apart by their training
configuration (evaluation-only switches such as the exit guard do not count), so the same model run in two
studies appears once: the run listed first is kept. Only models with all three seeds (42, 123, 7) are shown.
Machines are never mixed: one set of charts per device.

Writes docs/figures/all_models_<machine>_<attack>.png and docs/results/all_models.json.
Usage: python scripts/plot_all_models.py outputs
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rfl.experiment.config import load_config  # noqa: E402

_spec = importlib.util.spec_from_file_location("plot_results", Path(__file__).with_name("plot_results.py"))
assert _spec and _spec.loader
pr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pr)
plt = pr.plt

SEEDS = (42, 123, 7)
MACHINES = {"mps": "mac", "cuda": "gpu", "xpu": "acer"}
MACHINE_NAMES = {"mps": "Mac (MPS)", "cuda": "GPU server (CUDA)", "xpu": "Acer (XPU)"}
ATTACKS = [("clean", "No attack"), ("sign_flip", "Sign flip"), ("label_flip", "Label flip"),
           ("random_noise", "Random noise"), ("alie", "ALIE"), ("min_max", "Min-Max"), ("backdoor", "Backdoor"),
           ("stealth_backdoor", "Stealth backdoor")]
FAMILIES = [("Baselines (FedAvg, Multi-Krum)", "#9a9893"), ("Published FedHEP models", "#52514e"),
            ("Established robust rules in FedHEP (Phase 5)", pr.SLOT[0]),
            ("FedHEP+ and its variants (Phases 3-4)", pr.SLOT[1]),
            ("L0, R2, R3 and the screen variants (Phase 6)", pr.SLOT[2]), ("FedHEP-Guard (union screen)", pr.SLOT[3])]
EVAL_ONLY = {("fedhep", "exit_guard"), ("fedhep", "head_agreement"), ("run", "eval_every"), ("run", "output_dir"),
             ("run", "device"), ("run", "seed"), ("run", "deterministic")}


def condition(cfg: dict) -> str | None:
    a = cfg.get("attack") or {}
    kind = a.get("kind")
    if kind is None:
        return "clean"
    if a.get("count") != 3 or a.get("placement", "random") != "random" or (a.get("scale") or 1.0) != 1.0 \
            or (a.get("stop_round") or 0) != 0:
        return None
    return kind


def model_key(cfg: dict) -> str:
    """The training configuration without the attack and the evaluation-only switches."""
    keep = {}
    for section, value in cfg.items():
        if section in ("name", "attack"):
            continue
        if isinstance(value, dict):
            keep[section] = {k: v for k, v in sorted(value.items()) if (section, k) not in EVAL_ONLY}
        else:
            keep[section] = value
    return json.dumps(keep, sort_keys=True)


def label(cfg: dict) -> tuple[str, int]:
    """(readable name, family index) from the configuration."""
    m = cfg["method"]
    e = cfg["train"]["local_epochs"]
    if m in ("fedavg", "multikrum"):
        return {"fedavg": "FedAvg", "multikrum": "Multi-Krum"}[m] + f" (E = {e})", 0
    if m == "fedhep":
        return "FedHEP" + (", grouping on weights" if cfg["fedhep"].get("similarity_on") == "weights" else ""), 1
    if m == "fedhep_defended":
        return "Defended FedHEP", 1
    r = cfg.get("robust") or {}
    if m == "fedhep_robust":
        rule = r.get("rule")
        if rule == "arc_nnm_tm":
            f = r.get("assumed_attackers", 3)
            return ("L0 + worker momentum" if r.get("momentum") else "L0") + (f", f = {f}" if f != 3 else ""), 4
        name = {"median": "Median", "trimmed_mean": "Trimmed mean", "bulyan": "Bulyan", "fltrust": "FLTrust",
                "flame": "FLAME" + (" (on updates)" if r.get("flame_on") == "updates" else "")}.get(rule, rule)
        return name, 2
    p = cfg.get("plus") or {}
    sig = {"root_head": "", "weights": ", whole update", "critical": ", critical layers", "sign_align": ", sign align",
           "probe_logits": ", probe logits", "head_gap": ", head gap", "weights_critical": ""}[p.get("signal", "root_head")]
    extra = []
    if not p.get("regroup", False):
        extra.append("no regroup")
    if p.get("global_f_fixed"):
        extra.append("f fixed")
    if p.get("global_rule") == "arc_nnm_tm" and p.get("global_f", 3) != 3:
        extra.append(f"f = {p['global_f']}")
    rw = p.get("rewind", "none")
    if rw != "none":
        extra.append({"trunk": "rewind trunk", "trunk_root": "rewind trunk + root", "all": "rewind all"}[rw])
    if p.get("global_rule") == "arc_nnm_tm":
        if p.get("signal") == "weights_critical":
            name = "FedHEP-Guard" if rw == "trunk_root" else "FedHEP-Guard screen (union, no rewind)"
            return name + "".join(f", {x}" for x in extra if not x.startswith("rewind")), 5
        gate = p.get("gap_gate", 0)
        name = "R2" if not gate else ("R3" if gate == 8 else f"R3, gate {gate:g}")
        return name + sig + "".join(f", {x}" for x in extra), 4
    # SCCF Global step: FedHEP+ and its Phase 3-4 variants
    head = "FedHEP+"
    if p.get("drill_down"):
        head = "Drill-down " + ("v2" if p.get("drill_mode") == "evidence" else "v1")
    parts = []
    if p.get("reference") != "median":
        parts.append("mean reference")
    if p.get("two_sided"):
        parts.append("two-sided, patience 3")
    if p.get("readmit"):
        parts.append("readmit")
    if p.get("max_excluded", 1.0) < 1.0:
        parts.append("cap")
    if p.get("population") == "all":
        parts.append("full population")
    if p.get("evidence_cut", 1.5) != 1.5:
        parts.append(f"isolation cut {p['evidence_cut']:g}")
    parts += [x for x in extra if x not in ("f fixed",)]
    return head + sig + (" (" + ", ".join(parts) + ")" if parts else ""), 3


def collect(outputs: Path) -> dict:
    """{device: {attack: {label: {"family", "runs", "acc", "asr"}}}}"""
    groups: dict = defaultdict(lambda: defaultdict(dict))   # device -> attack -> key -> {seed: metrics}
    order: dict = defaultdict(list)
    names: dict = {}
    for path in sorted(outputs.glob("*/metrics.json")):
        cfg_path = path.with_name("config.yaml")
        if not cfg_path.exists():
            continue
        try:   # fill keys added after the run with their defaults (which preserve the old behaviour)
            cfg = dataclasses.asdict(load_config(cfg_path))
        except (TypeError, ValueError, KeyError):
            continue
        d, r = cfg["data"], cfg["run"]
        if (d.get("num_clients"), d.get("alpha"), d.get("train_subset"), r.get("rounds")) != (15, 0.5, 10000, 30):
            continue
        att = condition(cfg)
        if att is None:
            continue
        m = json.loads(path.read_text())
        if m["seed"] not in SEEDS:
            continue
        dev = str(m.get("device"))
        key = model_key(cfg)
        groups[dev][att].setdefault(key, {})
        if m["seed"] not in groups[dev][att][key]:
            groups[dev][att][key][m["seed"]] = m
        if key not in names:
            names[key] = (label(cfg), path.parent.name.rsplit("_s", 1)[0])
            order[dev].append(key)
    out: dict = {}
    for dev, by_att in groups.items():
        out[dev] = {}
        for att, by_key in by_att.items():
            rows = {}
            for key, by_seed in by_key.items():
                if set(by_seed) != set(SEEDS):
                    continue
                (name, fam), run = names[key]
                ms = [by_seed[s] for s in SEEDS]
                if name in rows:   # same label, different configuration: keep both, tell them apart by run name
                    name = f"{name} [{run}]"
                rows[name] = {"family": fam, "run": run, "acc": [x["final"]["pooled"] for x in ms],
                              "asr": [x["history"][-1].get("asr") for x in ms]}
            if rows:
                out[dev][att] = rows
    return out


def chart(dev: str, att: str, title: str, rows: dict) -> None:
    backdoor = att in ("backdoor", "stealth_backdoor")
    items = sorted(rows.items(), key=lambda kv: sum(kv[1]["acc"]) / 3)
    n = len(items)
    fig, axes = plt.subplots(1, 2 if backdoor else 1, figsize=(15 if backdoor else 10.5, 1.2 + 0.32 * n),
                             sharey=True, squeeze=False, gridspec_kw={"width_ratios": [1.3, 1]} if backdoor else None)
    ys = range(n)
    panels = [(axes[0][0], "acc", "accuracy of honest clients (%), mean of 3 seeds")]
    if backdoor:
        panels.append((axes[0][1], "asr", "backdoor success (%), lower is better"))
    for ax, metric, xlabel in panels:
        mu = [sum(v[metric]) / 3 for _, v in items]
        sd = [pr.mean_sd(v[metric])[1] for _, v in items]
        colors = [FAMILIES[v["family"]][1] for _, v in items]
        ax.barh(list(ys), mu, xerr=sd, color=colors, height=0.72, capsize=2,
                error_kw={"elinewidth": 0.8, "ecolor": pr.INK2})
        for y, (m_, s_) in enumerate(zip(mu, sd, strict=True)):
            ax.text(m_ + s_ + 0.4, y, f"{m_:.1f}", va="center", fontsize=7.5)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_xlim(0, max(mu) * 1.12 + 2)
        ax.grid(axis="y", visible=False)
    axes[0][0].set_yticks(list(ys), [name for name, _ in items], fontsize=8)
    used = sorted({v["family"] for _, v in items})
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILIES[f][1]) for f in used]
    fig.legend(handles, [FAMILIES[f][0] for f in used], loc="lower center", ncols=min(3, len(used)), fontsize=8,
               frameon=False)
    fig.suptitle(f"{title}: every model tested, {MACHINE_NAMES[dev]}, D1 (15 clients, α = 0.5"
                 + (", 3 attackers)" if att != "clean" else ")"), fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05 if len(used) <= 3 else 0.08, 1, 0.97))
    pr.save(fig, f"all_models_{MACHINES[dev]}_{att}.png")


def main() -> None:
    pr.style()
    data = collect(Path(sys.argv[1]))
    for dev in ("mps", "cuda"):   # the Acer's 30-round D1 runs hold only L0 and the union screen (f study)
        if dev not in data:
            continue
        for att, title in ATTACKS:
            rows = data[dev].get(att, {})
            if len(rows) >= 4:
                chart(dev, att, title, rows)
    out = Path(__file__).resolve().parents[1] / "docs" / "results" / "all_models.json"
    out.write_text(json.dumps({MACHINES.get(d, d): v for d, v in data.items()}, indent=1))
    for dev in sorted(data):
        print(dev, {att: len(rows) for att, rows in data[dev].items()})


if __name__ == "__main__":
    main()
