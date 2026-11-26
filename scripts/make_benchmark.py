"""Write the benchmark sweeps (configs/sweeps/bench_*.yaml): every model and variant ever tested, under the same
eight conditions, in every setting ever tested, on one machine (the GPU server).

  python scripts/make_benchmark.py

One file per setting (FedHEP-server models) plus one file for FedAvg and Multi-Krum in every setting (they cannot
run ALIE, Min-Max or the stealth backdoor). 20% of the clients attack; the robust rules are told the true count.
Run names follow the large grid's (gg15_a05__sign3__r3w), so runs the grid already made are skipped, not rerun.
The model list here is also what scripts/plot_benchmark.py draws, in this order."""
import math
from pathlib import Path
from statistics import NormalDist

import yaml

ROOT = Path(__file__).resolve().parents[1]
FEDHEP, DEFENDED, FEDAVG = ("configs/cifar100_fedhep.yaml", "configs/cifar100_fedhep_defended.yaml",
                            "configs/cifar100_fedavg.yaml")

# Settings ever tested, in the order the GPU should run them: the report's two settings first (D1, D2), then the
# large grid's, then the follow-up and parity settings. (clients, alpha, label)
SETTINGS = [(15, 0.5, "D1"), (20, 0.2, "D2"), (30, 0.5, "30 clients, α 0.5"), (15, 0.1, "15 clients, α 0.1"),
            (30, 0.1, "30 clients, α 0.1"), (50, 0.1, "50 clients, α 0.1"), (50, 0.5, "50 clients, α 0.5"),
            (20, 0.5, "20 clients, α 0.5"), (15, 0.05, "15 clients, α 0.05 (paper)")]

_PLUS = {"config": DEFENDED, "method": "fedhep_plus", "plus.reference": "median"}
_R2 = {**_PLUS, "plus.regroup": True, "plus.two_sided": True, "plus.isolation_patience": 3,
       "plus.exclusion_patience": 3, "plus.global_rule": "arc_nnm_tm"}
_R3 = {**_R2, "plus.gap_gate": 8}
_ROBUST = {"config": DEFENDED, "method": "fedhep_robust"}

# name: (family, label, overrides). Family indexes FAMILIES in plot_benchmark.py. The grid's eight models keep the
# grid's names and settings exactly (fedhep, defended, plus, l0, r3, r3w, r3wc, flame).
MODELS = {
    # 0 baselines (FedAvg family: separate file, five conditions)
    "fedavg": (0, "FedAvg", {"config": FEDAVG}),
    "multikrum": (0, "Multi-Krum", {"config": FEDAVG, "method": "multikrum"}),
    # 1 the published models
    "fedhep": (1, "FedHEP", {"config": FEDHEP}),
    "fedhep_w": (1, "FedHEP, grouping on weights", {"config": FEDHEP, "fedhep.similarity_on": "weights"}),
    "defended": (1, "Defended FedHEP", {"config": DEFENDED}),
    # 2 FedHEP+ and its variants (Phases 3-4)
    "plus": (2, "FedHEP+", {**_PLUS, "plus.regroup": True}),
    "plus_mean": (2, "FedHEP+, mean reference, no regroup (first version)",
                  {"config": DEFENDED, "method": "fedhep_plus"}),
    "plus_nr": (2, "FedHEP+, no regroup", dict(_PLUS)),
    "plus_w": (2, "FedHEP+, whole update, no regroup", {**_PLUS, "plus.signal": "weights"}),
    "plus_ra": (2, "FedHEP+, readmit", {**_PLUS, "plus.regroup": True, "plus.readmit": True}),
    "plus_ra_cap": (2, "FedHEP+, readmit, cap",
                    {**_PLUS, "plus.regroup": True, "plus.readmit": True, "plus.max_excluded": 0.34}),
    "plus_ra_cap_nr": (2, "FedHEP+, readmit, cap, no regroup",
                       {**_PLUS, "plus.readmit": True, "plus.max_excluded": 0.34}),
    "plus_full": (2, "FedHEP+, full population", {**_PLUS, "plus.regroup": True, "plus.population": "all"}),
    "plus_full_nr": (2, "FedHEP+, full population, no regroup", {**_PLUS, "plus.population": "all"}),
    "plus_full_nr_cut2": (2, "FedHEP+, full population, isolation cut 2, no regroup",
                          {**_PLUS, "plus.population": "all", "plus.evidence_cut": 2.0}),
    "plus_p3": (2, "FedHEP+, patience 3", {**_PLUS, "plus.regroup": True, "plus.isolation_patience": 3,
                                            "plus.exclusion_patience": 3}),
    "plus_two": (2, "FedHEP+, two-sided, patience 3",
                 {**_PLUS, "plus.regroup": True, "plus.two_sided": True, "plus.isolation_patience": 3,
                  "plus.exclusion_patience": 3}),
    "plus_floor001": (2, "FedHEP+, scale floor 0.01", {**_PLUS, "plus.regroup": True, "plus.mad_floor": 0.01}),
    "plus_floor003": (2, "FedHEP+, scale floor 0.03", {**_PLUS, "plus.regroup": True, "plus.mad_floor": 0.03}),
    "drill1": (2, "Drill-down v1", {**_PLUS, "plus.drill_down": True}),
    "drill2": (2, "Drill-down v2", {**_PLUS, "plus.drill_down": True, "plus.drill_mode": "evidence",
                                     "plus.min_group": 3}),
    "drill2_all": (2, "Drill-down v2, every group checked",
                   {**_PLUS, "plus.drill_down": True, "plus.drill_mode": "evidence", "plus.min_group": 3,
                    "plus.group_cut": -1.0e9}),
    # 3 established robust rules in FedHEP's Global step (Phase 5)
    "median": (3, "Median", {**_ROBUST, "robust.rule": "median"}),
    "tmean": (3, "Trimmed mean", {**_ROBUST, "robust.rule": "trimmed_mean"}),
    "bulyan": (3, "Bulyan", {**_ROBUST, "robust.rule": "bulyan"}),
    "flame": (3, "FLAME", {**_ROBUST, "robust.rule": "flame"}),
    "flame_upd": (3, "FLAME on updates", {**_ROBUST, "robust.rule": "flame", "robust.flame_on": "updates"}),
    "fltrust": (3, "FLTrust", {**_ROBUST, "robust.rule": "fltrust"}),
    # 4 the FedHEP-Guard ladder and its variants (Phases 6-8)
    "l0": (4, "L0", {**_ROBUST, "robust.rule": "arc_nnm_tm"}),
    "l0m": (4, "L0 + worker momentum", {**_ROBUST, "robust.rule": "arc_nnm_tm", "robust.momentum": 0.5}),
    "r2": (4, "R2", dict(_R2)),
    "r3": (4, "R3 gate 8, root head", dict(_R3)),
    "r3g12": (4, "R3 gate 12, root head", {**_R2, "plus.gap_gate": 12}),
    "r3_nr": (4, "R3, no regroup", {**_R3, "plus.regroup": False}),
    "r3_ffixed": (4, "R3, f fixed", {**_R3, "plus.global_f_fixed": True}),
    "r3_nr_ffixed": (4, "R3, no regroup, f fixed", {**_R3, "plus.regroup": False, "plus.global_f_fixed": True}),
    "r3w": (4, "R3, whole update", {**_R3, "plus.signal": "weights"}),
    "r3c": (4, "R3, critical layers", {**_R3, "plus.signal": "critical"}),
    "r3s": (4, "R3, sign alignment", {**_R3, "plus.signal": "sign_align"}),
    "r3p": (4, "R3, probe logits", {**_R3, "plus.signal": "probe_logits"}),
    "r3h": (4, "R3, head gap", {**_R3, "plus.signal": "head_gap"}),
    "r3_trunk": (4, "R3, rewind trunk", {**_R3, "plus.rewind": "trunk"}),
    "r3_tr": (4, "R3, rewind trunk + root", {**_R3, "plus.rewind": "trunk_root"}),
    "r3_all": (4, "R3, rewind all", {**_R3, "plus.rewind": "all"}),
    "r3w_trunk": (4, "R3, whole update, rewind trunk", {**_R3, "plus.signal": "weights", "plus.rewind": "trunk"}),
    "r3w_tr": (4, "R3, whole update, rewind trunk + root",
               {**_R3, "plus.signal": "weights", "plus.rewind": "trunk_root"}),
    "r3w_all": (4, "R3, whole update, rewind all", {**_R3, "plus.signal": "weights", "plus.rewind": "all"}),
    "peer": (4, "Union screen, peer reference (Phase 7)",
             {**_R3, "plus.signal": "weights_critical", "plus.reference": "peer"}),
    "shared": (4, "Union screen, shared signal (Phase 8)", {**_R3, "plus.signal": "shared_critical"}),
    "guard_shared": (4, "FedHEP-Guard, shared signal (Phase 8)",
                     {**_R3, "plus.signal": "shared_critical", "plus.rewind": "trunk_root"}),
    # 5 FedHEP-Guard: the union screen without and with the rewind
    "r3wc": (5, "FedHEP-Guard without rewind (union screen)", {**_R3, "plus.signal": "weights_critical"}),
    "guard": (5, "FedHEP-Guard", {**_R3, "plus.signal": "weights_critical", "plus.rewind": "trunk_root"}),
    # the low-side gate at 6 and the high side at 20, chosen in the offline replay (report 8.2), not yet run live
    "r3wc_g6h20": (5, "FedHEP-Guard without rewind, gate 6, high side 20",
                   {**_R3, "plus.signal": "weights_critical", "plus.gap_gate": 6, "plus.high_gate": 20}),
    "guard_g6h20": (5, "FedHEP-Guard, gate 6, high side 20",
                    {**_R3, "plus.signal": "weights_critical", "plus.gap_gate": 6, "plus.high_gate": 20,
                     "plus.rewind": "trunk_root"}),
}
# Models also written to a short GPU sweep that runs them in D1 and D2 first (same run names as the benchmark).
FIRST = {"gpu_c8_gates.yaml": ("r3wc_g6h20", "guard_g6h20")}
FEDAVG_FAMILY = ("fedavg", "multikrum")


def tag(n: int, alpha: float) -> str:
    return f"gg{n}_a" + f"{alpha:g}".replace("0.", "0")


def alie_z(n: int, m: int) -> float:
    """Baruch et al. (2019): s = floor(n/2 + 1) - m supporters needed, z = Phi^-1((n - s) / n)."""
    return round(NormalDist().inv_cdf((n - math.floor(n / 2 + 1) + m) / n), 3)


def conditions(n: int, fedavg: bool) -> dict:
    m = round(0.2 * n)
    f = {"attack.count": m, "robust.assumed_attackers": m, "plus.global_f": m}
    out = {"clean": {"robust.assumed_attackers": m, "plus.global_f": m},
           f"sign{m}": {"attack.kind": "sign_flip", **f},
           f"label{m}": {"attack.kind": "label_flip", **f},
           f"noise{m}": {"attack.kind": "random_noise", **f},
           f"alie{m}": {"attack.kind": "alie", "attack.knowledge": "colluders", "attack.alie_z": alie_z(n, m), **f},
           f"minmax{m}": {"attack.kind": "min_max", "attack.knowledge": "omniscient", **f},
           f"backdoor{m}": {"attack.kind": "backdoor", **f},
           f"stealth{m}": {"attack.kind": "stealth_backdoor", **f}}
    if fedavg:   # FedAvg and Multi-Krum have no FedHEP server: no ALIE, Min-Max or stealth backdoor
        out = {k: {kk: vv for kk, vv in v.items() if kk == "attack.count" or kk.startswith("attack.kind")}
               for k, v in out.items() if not k.startswith(("alie", "minmax", "stealth"))}
    return out


COMMON = {"run.deterministic": True, "run.rounds": 30, "fedhep.exit_guard": True, "fedhep.head_agreement": True}


def main() -> None:
    out = ROOT / "configs" / "sweeps"
    fedhep_models = {k: v[2] for k, v in MODELS.items() if k not in FEDAVG_FAMILY}
    total = 0
    for i, (n, alpha, label) in enumerate(SETTINGS, 1):
        spec = {"seeds": [42, 123, 7], "common": {**COMMON, "data.num_clients": n},
                "axes": [{tag(n, alpha): {"data.alpha": alpha}}, conditions(n, False), fedhep_models]}
        runs = 3 * len(spec["axes"][1]) * len(fedhep_models)
        total += runs
        head = (f"# Benchmark, setting {i} of {len(SETTINGS)}: {label} ({n} clients, alpha {alpha}, "
                f"{round(0.2 * n)} attackers). Written by scripts/make_benchmark.py; edit that, not this file.\n"
                f"# {len(fedhep_models)} models x 8 conditions x 3 seeds = {runs} runs. GPU server only. Runs the "
                "large grid already made (same names) are skipped.\n")
        (out / f"bench_{i}_{tag(n, alpha)}.yaml").write_text(head + yaml.safe_dump(spec, sort_keys=False,
                                                                                    allow_unicode=True))
    spec = {"seeds": [42, 123, 7], "common": {"run.deterministic": True, "run.rounds": 30},
            "axes": [{tag(n, a): {"data.num_clients": n, "data.alpha": a, "attack.count": round(0.2 * n)}
                      for n, a, _ in SETTINGS},
                     {k: {kk: vv for kk, vv in v.items() if kk != "attack.count"}
                      for k, v in conditions(15, True).items()},
                     {k: MODELS[k][2] for k in FEDAVG_FAMILY}]}
    # the attack count depends on the setting, so it sits on the setting axis; condition labels drop the count
    spec["axes"][1] = {k.rstrip("0123456789"): v for k, v in spec["axes"][1].items()}
    runs = 3 * len(SETTINGS) * len(spec["axes"][1]) * len(FEDAVG_FAMILY)
    total += runs
    head = ("# Benchmark, FedAvg and Multi-Krum in every setting (no FedHEP server: no ALIE, Min-Max or stealth "
            "backdoor). Written by scripts/make_benchmark.py.\n"
            f"# {len(SETTINGS)} settings x 5 conditions x 2 models x 3 seeds = {runs} runs. GPU server only.\n")
    (out / "bench_0_fedavg_family.yaml").write_text(head + yaml.safe_dump(spec, sort_keys=False, allow_unicode=True))
    for fname, models in FIRST.items():
        spec = {"seeds": [42, 123, 7], "common": COMMON,
                "axes": [{f"{tag(n, a)}__{c}": {"data.num_clients": n, "data.alpha": a, **v}
                          for n, a, _ in SETTINGS[:2] for c, v in conditions(n, False).items()},
                         {k: MODELS[k][2] for k in models}]}
        head = (f"# GPU server: {', '.join(models)} in D1 and D2 before the benchmark reaches them. Written by "
                f"scripts/make_benchmark.py. Same run names as bench_1/bench_2, so whichever runs first is kept.\n"
                f"# 2 settings x 8 conditions x {len(models)} models x 3 seeds = {48 * len(models)} runs.\n")
        (out / fname).write_text(head + yaml.safe_dump(spec, sort_keys=False, allow_unicode=True))
    print(f"{len(SETTINGS)} settings, {len(MODELS)} models, {total} runs in total")


if __name__ == "__main__":
    main()
