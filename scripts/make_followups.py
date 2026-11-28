"""Write the follow-up sweeps of the literature review, Part 2 (docs/LITERATURE_REVIEW.md, experiments 2 and 3).

  python scripts/make_followups.py

gpu_c9_attack_strength.yaml (experiment 3): the benchmark's ALIE and Min-Max are bounded by design (ALIE's z from
Baruch et al.'s formula, 0.43 at D1; Min-Max along the inverse std) and barely hurt FedHEP. Here every model faces
damage-oriented settings from the literature: ALIE at z = 1, 1.5 (Baruch et al.'s own experiments) and 3; IPM (Xie
et al.) at epsilon 0.5, 2 and the number of clients; Min-Max along the inverse unit vector and the inverse sign.
gpu_c10_assumed_f.yaml (experiment 2): the robust global step assumes 20% attackers even with no attack; here it
assumes none, to split the no-attack cost between the global step, the screen and the rewind.
gpu_c12_fixed_f.yaml (Part 4): the global step keeps assuming 20% attackers as clients are isolated, at 50 clients.
gpu_c11_attack_suite.yaml (Part 3): Min-Sum, Fang's trimmed-mean attack, PoisonedFL, XFED and three attack schedules
(2-round bursts, which never give the screen 3 bad rounds in a row, and a sleeper).
gpu_c12_fixed_f.yaml also holds a floor on f (plus.global_f_floor 0.5: f never below half the starting count).
gpu_c13_fixes.yaml and gpu_c13b_schedules.yaml (report 8.2, 2026-11-28): fixes for the failures found so far, each
alone and together: the rewind only for two-signal isolations (plus.rewind_agree), isolation after 3 bad rounds in
any 5 instead of in a row (plus.isolation_window), the low-side gate of 6 with the high side at 20, and the f floor.
gpu_c14_mixed.yaml (Part 4, protocol M1-M3, M6): mixed attacker populations in one run (attack.kind mixed).

Both use the benchmark's settings, models and run names (scripts/make_benchmark.py), GPU server only."""
import importlib.util
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("make_benchmark", Path(__file__).with_name("make_benchmark.py"))
mb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mb)

STRENGTH_MODELS = ["fedhep", "defended", "plus", "r3wc", "guard", "flame", "bulyan", "median"]


def strength_conditions(n: int) -> dict:
    m = round(0.2 * n)
    f = {"attack.count": m, "robust.assumed_attackers": m, "plus.global_f": m}
    out = {}
    for z in (1, 1.5, 3):
        out[f"alie{m}_z{z:g}"] = {"attack.kind": "alie", "attack.knowledge": "colluders", "attack.alie_z": z, **f}
    for e in (0.5, 2, n):
        out[f"ipm{m}_e{e:g}"] = {"attack.kind": "ipm", "attack.knowledge": "omniscient", "attack.ipm_epsilon": e, **f}
    for p in ("unit", "sign"):
        out[f"minmax{m}_{p}"] = {"attack.kind": "min_max", "attack.knowledge": "omniscient",
                                 "attack.perturbation": p, **f}
    return out


def suite_conditions(n: int) -> dict:
    """Attacks added after the second literature round (Part 3) and the attack schedules."""
    m = round(0.2 * n)
    f = {"attack.count": m, "robust.assumed_attackers": m, "plus.global_f": m}
    return {f"minsum{m}": {"attack.kind": "min_sum", "attack.knowledge": "omniscient", **f},
            f"fang{m}_full": {"attack.kind": "fang_trim", "attack.knowledge": "omniscient", **f},
            f"fang{m}_partial": {"attack.kind": "fang_trim", "attack.knowledge": "colluders", **f},
            f"poisonedfl{m}": {"attack.kind": "poisonedfl", **f},
            f"xfed{m}": {"attack.kind": "xfed", **f},
            f"sign{m}_burst21": {"attack.kind": "sign_flip", "attack.on_rounds": 2, "attack.off_rounds": 1, **f},
            f"sign{m}_sleeper10": {"attack.kind": "sign_flip", "attack.start_round": 10, **f},
            f"backdoor{m}_burst21": {"attack.kind": "backdoor", "attack.on_rounds": 2, "attack.off_rounds": 1, **f}}


# the global step's assumed attacker count kept at 20% of the clients however many are isolated (plus.global_f_fixed):
# by default f = global_f - |isolated|, which reaches 0 (a plain mean) at 50 clients by rounds 6-12 with 4-8 of the 10
# backdoor attackers still in (team logs, 2026-11-27)
FIXED_F = {"r3wc_ffixed": {**mb.MODELS["r3wc"][2], "plus.global_f_fixed": True},
           "guard_ffixed": {**mb.MODELS["guard"][2], "plus.global_f_fixed": True},
           # f never below half the starting count (literature review Part 4, experiment 2)
           "r3wc_ffloor": {**mb.MODELS["r3wc"][2], "plus.global_f_floor": 0.5},
           "guard_ffloor": {**mb.MODELS["guard"][2], "plus.global_f_floor": 0.5}}

# Fixes for the failures found so far (report 8.2). guard_v2 is every fix together.
FIXES = {"guard_agree": {**mb.MODELS["guard"][2], "plus.rewind_agree": True},
         "r3wc_w5g6h20": {**mb.MODELS["r3wc_g6h20"][2], "plus.isolation_window": 5},
         "guard_v2": {**mb.MODELS["guard_g6h20"][2], "plus.isolation_window": 5, "plus.rewind_agree": True,
                      "plus.global_f_floor": 0.5}}
SCHEDULE_MODELS = {k: mb.MODELS[k][2] for k in ("r3wc", "guard", "r3wc_g6h20")} | FIXES
MIXED_MODELS = {k: mb.MODELS[k][2] for k in ("fedhep", "plus_w", "flame", "r3wc", "guard")} | {
    "guard_v2": FIXES["guard_v2"]}


def schedule_conditions(n: int) -> dict:
    m = round(0.2 * n)
    f = {"attack.count": m, "robust.assumed_attackers": m, "plus.global_f": m}
    return {f"backdoor{m}_burst21": {"attack.kind": "backdoor", "attack.on_rounds": 2, "attack.off_rounds": 1, **f},
            f"backdoor{m}_sleeper10": {"attack.kind": "backdoor", "attack.start_round": 10, **f}}


def mixed_conditions(n: int) -> dict:
    """Literature review Part 4, protocol table: M1 masking, M2 bounded cover, M3 staggered decoys, M6
    non-cooperative. Half-and-half mixes; M3 has one decoy per 15 clients (rounded down), decoy k from round 3 + 5k."""
    m = round(0.2 * n)
    h, decoys = m // 2, max(1, n // 15)
    f = {"attack.kind": "mixed", "robust.assumed_attackers": m, "plus.global_f": m}
    stealth = {"kind": "stealth_backdoor"}
    return {f"m1_{m}": {**f, "attack.mix": [{**stealth, "count": h}, {"kind": "sign_flip", "count": m - h}]},
            f"m2_{m}": {**f, "attack.mix": [{**stealth, "count": h},
                                            {"kind": "alie", "knowledge": "colluders", "alie_z": 1.5,
                                             "count": m - h}]},
            f"m3_{m}": {**f, "attack.mix": [{**stealth, "count": m - decoys}] + [
                {"kind": "sign_flip", "count": 1, "start_round": 3 + 5 * k} for k in range(decoys)]},
            f"m6_{m}": {**f, "attack.mix": [{"kind": "backdoor", "count": h}, {"kind": "xfed", "count": m - h}]}}


# the global step's assumed attacker count set to 0, otherwise the benchmark models
ASSUMED_F = {"l0_f0": {**mb.MODELS["l0"][2], "robust.assumed_attackers": 0},
             "r3wc_f0": {**mb.MODELS["r3wc"][2], "plus.global_f": 0},
             "guard_f0": {**mb.MODELS["guard"][2], "plus.global_f": 0}}


def setting_axis(settings, conds) -> dict:
    return {f"{mb.tag(n, a)}__{c}": {"data.num_clients": n, "data.alpha": a, **v}
            for n, a, _ in settings for c, v in conds(n).items()}


def main() -> None:
    out = ROOT / "configs" / "sweeps"
    spec = {"seeds": [42, 123, 7], "common": mb.COMMON,
            "axes": [setting_axis(mb.SETTINGS[:2], strength_conditions),
                     {k: mb.MODELS[k][2] for k in STRENGTH_MODELS}]}
    runs = 3 * len(spec["axes"][0]) * len(STRENGTH_MODELS)
    head = ("# GPU server: attack strength (literature review Part 2, experiment 3; predictions P44-P47). D1 and D2; "
            "ALIE z 1 / 1.5 / 3 (colluders), IPM epsilon 0.5 / 2 / n (omniscient), Min-Max along the inverse unit "
            "vector and the inverse sign (omniscient).\n# Written by scripts/make_followups.py. "
            f"2 settings x 8 attacks x {len(STRENGTH_MODELS)} models x 3 seeds = {runs} runs. The benchmark holds "
            "the bounded versions (formula z, inverse std) and the no-attack runs.\n")
    (out / "gpu_c9_attack_strength.yaml").write_text(head + yaml.safe_dump(spec, sort_keys=False, allow_unicode=True))

    spec = {"seeds": [42, 123, 7], "common": mb.COMMON,
            "axes": [setting_axis(mb.SETTINGS[:2], suite_conditions),
                     {k: mb.MODELS[k][2] for k in STRENGTH_MODELS}]}
    runs = 3 * len(spec["axes"][0]) * len(STRENGTH_MODELS)
    head = ("# GPU server: the rest of the attack suite and the attack schedules (literature review Part 3; "
            "predictions P57-P61). D1 and D2; Min-Sum, Fang's trimmed-mean attack (full and partial knowledge), "
            "PoisonedFL, XFED;\n# sign flip in 2-round bursts (2 on, 1 off) and from round 10 (sleeper); backdoor in "
            f"2-round bursts. Written by scripts/make_followups.py. 2 settings x 8 conditions x {len(STRENGTH_MODELS)} "
            f"models x 3 seeds = {runs} runs.\n")
    (out / "gpu_c11_attack_suite.yaml").write_text(head + yaml.safe_dump(spec, sort_keys=False, allow_unicode=True))

    settings = [mb.SETTINGS[i] for i in (0, 1, 2, 5)]   # D1, D2, 30 clients alpha 0.5, 50 clients alpha 0.1

    def fix_conditions(n: int) -> dict:
        c = mb.conditions(n, False)
        m = round(0.2 * n)
        return {k: c[k] for k in ("clean", f"sign{m}", f"backdoor{m}", f"stealth{m}", f"minmax{m}")}

    spec = {"seeds": [42, 123, 7], "common": mb.COMMON, "axes": [setting_axis(settings, fix_conditions), FIXES]}
    runs = 3 * len(spec["axes"][0]) * len(FIXES)
    head = ("# GPU server: fixes for the failures found so far (report 8.2; predictions P70-P72). D1, D2, 30 clients "
            "alpha 0.5, 50 clients alpha 0.1; no attack, sign flip, backdoor, stealth backdoor, Min-Max.\n# guard_agree "
            "(rewind only for two-signal isolations), r3wc_w5g6h20 (3 bad rounds in any 5, gate 6, high side 20), "
            "guard_v2 (both, and f never below half the starting count). Controls: the benchmark's r3wc, guard, "
            f"r3wc_g6h20, guard_g6h20 (same run names). Written by scripts/make_followups.py. {runs} runs.\n")
    (out / "gpu_c13_fixes.yaml").write_text(head + yaml.safe_dump(spec, sort_keys=False, allow_unicode=True))

    spec = {"seeds": [42, 123, 7], "common": mb.COMMON,
            "axes": [setting_axis(settings, schedule_conditions), SCHEDULE_MODELS]}
    runs = 3 * len(spec["axes"][0]) * len(SCHEDULE_MODELS)
    head = ("# GPU server: the timed attacks that beat every isolation rule on the Acer (report 8.2), at four "
            "settings: backdoor 2 rounds on, 1 off; backdoor from round 10.\n# Controls r3wc, guard, r3wc_g6h20 in "
            "the same sweep; fixes guard_agree, r3wc_w5g6h20, guard_v2 (P72). Written by scripts/make_followups.py. "
            f"{runs} runs.\n")
    (out / "gpu_c13b_schedules.yaml").write_text(head + yaml.safe_dump(spec, sort_keys=False, allow_unicode=True))

    spec = {"seeds": [42, 123, 7], "common": mb.COMMON,
            "axes": [setting_axis([mb.SETTINGS[1], mb.SETTINGS[2]], mixed_conditions), MIXED_MODELS]}
    runs = 3 * len(spec["axes"][0]) * len(MIXED_MODELS)
    head = ("# GPU server: mixed attacker populations (literature review Part 4; predictions P73-P74). D2 and 30 "
            "clients alpha 0.5; M1 stealth backdoor + sign flip, M2 stealth backdoor + ALIE z 1.5,\n# M3 stealth "
            "backdoor + sign-flip decoys starting at rounds 3 (and 8), M6 backdoor + XFED. Single-attack references: "
            f"the benchmark. Written by scripts/make_followups.py. {runs} runs.\n")
    (out / "gpu_c14_mixed.yaml").write_text(head + yaml.safe_dump(spec, sort_keys=False, allow_unicode=True))

    def fixed_conditions(n: int) -> dict:
        c = mb.conditions(n, False)
        m = round(0.2 * n)
        return {k: c[k] for k in ("clean", f"backdoor{m}", f"stealth{m}")}

    spec = {"seeds": [42, 123, 7], "common": mb.COMMON,
            "axes": [setting_axis([s for s in mb.SETTINGS if s[0] == 50], fixed_conditions), FIXED_F]}
    runs = 3 * len(spec["axes"][0]) * len(FIXED_F)
    head = ("# GPU server: the global step's assumed attacker count held at 20% of the clients however many are isolated "
            "(plus.global_f_fixed; literature review Part 4; predictions P62-P63), or never below half of it "
            "(plus.global_f_floor 0.5; P75).\n# 50 clients, alpha 0.1 and 0.5; no attack, backdoor, stealth backdoor; "
            "union screen and FedHEP-Guard. The benchmark holds the same models with the shrinking count. Written by "
            f"scripts/make_followups.py. {runs} runs.\n")
    (out / "gpu_c12_fixed_f.yaml").write_text(head + yaml.safe_dump(spec, sort_keys=False, allow_unicode=True))

    def f_conditions(n: int) -> dict:
        c = mb.conditions(n, False)
        m = round(0.2 * n)
        return {k: c[k] for k in ("clean", f"sign{m}", f"backdoor{m}")}

    spec = {"seeds": [42, 123, 7], "common": mb.COMMON,
            "axes": [setting_axis(mb.SETTINGS[:3], f_conditions), ASSUMED_F]}
    runs = 3 * len(spec["axes"][0]) * len(ASSUMED_F)
    head = ("# GPU server: the global step's assumed attacker count set to 0 (literature review Part 2, experiment 2; "
            "predictions P48-P50). D1, D2 and 30 clients alpha 0.5; no attack, sign flip, backdoor;\n# L0, the "
            "union screen and FedHEP-Guard. The benchmark holds the same models with the true count assumed. Written "
            f"by scripts/make_followups.py. 3 settings x 3 conditions x 3 models x 3 seeds = {runs} runs.\n")
    (out / "gpu_c10_assumed_f.yaml").write_text(head + yaml.safe_dump(spec, sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()
