"""Write the large GPU grid (configs/sweeps/gpu_grid_n{15,30,50}.yaml): one file per client count, because the
attacker count, the f the robust rules assume and the ALIE z all depend on n and m.

  python scripts/make_gpu_grid.py

ALIE z follows Baruch et al. (2019): s = floor(n/2 + 1) - m supporters needed, z = Phi^-1((n - s) / n)."""
import math
from pathlib import Path
from statistics import NormalDist

import yaml

ROOT = Path(__file__).resolve().parents[1]
SHARES = {15: (2, 3, 5), 30: (3, 6, 9), 50: (5, 10, 15)}   # about 10 / 20 / 30% attackers
DEFENDED = "configs/cifar100_fedhep_defended.yaml"
R3 = {"config": DEFENDED, "method": "fedhep_plus", "plus.reference": "median", "plus.regroup": True,
      "plus.two_sided": True, "plus.isolation_patience": 3, "plus.exclusion_patience": 3,
      "plus.global_rule": "arc_nnm_tm", "plus.gap_gate": 8}


def alie_z(n: int, m: int) -> float:
    s = math.floor(n / 2 + 1) - m
    return round(NormalDist().inv_cdf((n - s) / n), 3)


def spec(n: int) -> dict:
    methods = {
        "fedhep": {"config": "configs/cifar100_fedhep.yaml"},
        "defended": {"config": DEFENDED},
        "plus": {"config": DEFENDED, "method": "fedhep_plus", "plus.reference": "median", "plus.regroup": True},
        "l0": {"config": DEFENDED, "method": "fedhep_robust", "robust.rule": "arc_nnm_tm"},
        "r3": dict(R3),
        "r3w": {**R3, "plus.signal": "weights"},
        "r3wc": {**R3, "plus.signal": "weights_critical"},
        "flame": {"config": DEFENDED, "method": "fedhep_robust", "robust.rule": "flame"},
    }
    alphas = {f"gg{n}_a01": {"data.alpha": 0.1}, f"gg{n}_a05": {"data.alpha": 0.5}}
    mid = SHARES[n][1]
    # f known to the defender = the true attacker count (as in every earlier table); clean runs assume the middle
    # share.
    attacks = {"clean": {"robust.assumed_attackers": mid, "plus.global_f": mid}}
    for m in SHARES[n]:
        f = {"attack.count": m, "robust.assumed_attackers": m, "plus.global_f": m}
        attacks[f"sign{m}"] = {"attack.kind": "sign_flip", **f}
        attacks[f"alie{m}"] = {"attack.kind": "alie", "attack.knowledge": "colluders",
                               "attack.alie_z": alie_z(n, m), **f}
        attacks[f"minmax{m}"] = {"attack.kind": "min_max", "attack.knowledge": "omniscient", **f}
        attacks[f"backdoor{m}"] = {"attack.kind": "backdoor", **f}
    return {"seeds": [42, 123, 7],
            "common": {"run.deterministic": True, "data.num_clients": n, "run.rounds": 30,
                       "fedhep.exit_guard": True, "fedhep.head_agreement": True},
            # Methods innermost: runs go out cell by cell (all 8 methods x 3 seeds of one setting together), so
            # a stopped or half-finished job still leaves complete comparisons. Names: gg15_a05__sign3__r3w.
            "axes": [alphas, attacks, methods]}


def main() -> None:
    for n in SHARES:
        s = spec(n)
        runs = len(s["seeds"]) * math.prod(len(a) for a in s["axes"])
        head = (f"# Large GPU grid, {n} clients (written by scripts/make_gpu_grid.py; edit that, not this file).\n"
                f"# 8 methods x alpha 0.1 / 0.5 x (clean + 4 attacks x {len(SHARES[n])} attacker counts "
                f"{SHARES[n]}) x 3 seeds = {runs} runs. GPU server only.\n")
        path = ROOT / f"configs/sweeps/gpu_grid_n{n}.yaml"
        path.write_text(head + yaml.safe_dump(s, sort_keys=False, width=120))
        print(path.relative_to(ROOT), runs, {m: alie_z(n, m) for m in SHARES[n]})


if __name__ == "__main__":
    main()
