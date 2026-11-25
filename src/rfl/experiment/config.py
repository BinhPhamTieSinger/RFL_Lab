"""The single place where experiment settings and their defaults are declared.

A YAML file overrides any subset. Unknown keys are an error, so a typo can't silently fall back to a default.
Defaults match the authors' CIFAR-100 Table 2 setup; the source of each is noted beside it."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from rfl.client.sgd import SGDSettings


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "cifar100"            # "cifar100" or "synthetic" (tests only)
    root: str = "./data"
    train_subset: int | None = 10000     # authors' CIFAR100_DEFAULTS
    test_subset: int | None = 3000       # authors' CIFAR100_DEFAULTS
    num_clients: int = 15                # paper and code
    alpha: float | None = None           # Dirichlet alpha; None = IID. Paper regimes: None, 1.0, 0.5, 0.1, 0.05


@dataclass(frozen=True)
class RunConfig:
    rounds: int = 25                     # authors' recorded Table 2 runs (their runner's default is 20)
    eval_every: int = 5                  # authors' eval_interval; the last round is always evaluated
    seed: int = 42
    device: str = "auto"                 # auto: cuda, then xpu, then mps, then cpu
    output_dir: str = "outputs"
    deterministic: bool = False          # deterministic GPU kernels: a rerun on the same device type gives the
                                         # same numbers (cuDNN and atomics otherwise reorder float additions)


@dataclass(frozen=True)
class AttackConfig:
    kind: str | None = None              # None, "label_flip", "sign_flip", "gradient_ascent", "random_noise",
                                         # "backdoor", "alie", "min_max", "stealth_backdoor" (FedHEP servers:
                                         # backdoor blended toward the honest mean, rfl.attacks.adaptive)
    knowledge: str = "colluders"       # coordinated attacks: colluders | omniscient
    alie_z: float = 1.0
    rate: float = 0.0                    # each client is an attacker with this probability (authors' rule)
    count: int | None = None             # exactly this many attackers instead (overrides rate)
    stop_round: int = 0                  # FedHEP family: attackers train honestly after this round (0 = never)
    placement: str = "random"            # grouped methods' initial groups: "random" (as drawn), "concentrated"
                                         # (all attackers in parent 0) or "spread" (attackers dealt round-robin)
    backdoor_target: int = 0
    backdoor_fraction: float = 0.5       # share of an attacker's training images stamped and relabelled
    scale: float = 1.0                   # backdoor upload boost; > 1 is model replacement


@dataclass(frozen=True)
class FedHEPConfig:
    num_groups: int = 3                  # authors' num_clusters (parents)
    similarity_on: str = "full"          # what grouping compares: "full" flat update (authors) or "weights" only
    exit_guard: bool = False             # evaluation only: also score with the depth-exit guard (rfl.defense.
                                         # exit_guard); the main metrics stay the unguarded ones
    head_agreement: bool = False         # evaluation only: also score with head-agreement inference (idea 9:
                                         # the personal heads overrule the ensemble when they agree against it)


@dataclass(frozen=True)
class PlusConfig:
    """FedHEP+ (Phase 3); see rfl.defense.screening."""
    signal: str = "root_head"            # "root_head" (SCCF's signal) | "weights" (whole update minus BatchNorm)
                                         # | "critical" (most disputed layers) | "sign_align" (AlignIns MPSA)
                                         # | "probe_logits" | "head_gap" (outputs on a server probe set)
                                         # | "weights_critical" (either screen can flag: lower of the two z)
                                         # | "shared" (whole update without the personal heads, Phase 8)
                                         # | "shared_critical" (union of "shared" and "critical")
    probe_size: int = 200                # probe images for the output signals (FLTrust-style root set)
    critical_k: int = 5                  # "critical" signal: most disputed shared layers screened per round
    sign_topk: float = 0.3               # "sign_align" signal: share of largest coordinates compared (AlignIns)
    reference: str = "mean_direction"    # "mean_direction" (SCCF's) | "median" | "peer" (Phase 7: each client
                                         # against the median of the clients holding the most similar classes)
    peer_share: float = 0.333            # "peer": share of the other clients used as a client's peers
    shadow_reference: str = ""           # also score against this reference and log it, without acting
    high_gate: float = 0.0               # two-sided: z far ABOVE the others counts only in rounds whose gap ratio
                                         # reaches this (0: same as gap_gate). Report 8.2: backdoor attackers sit
                                         # low; colluding ALIE / Min-Max high, in rounds with gap ratios of 30-130
    z_cut: float = 2.5                   # leave a client out this round when its robust z < -z_cut
    evidence_cut: float = 1.5            # isolate for good when its running z falls below -evidence_cut
    evidence_decay: float = 0.7
    regroup: bool = False                # regroup clients by weight-update similarity (else random groups)
    readmit: bool = False                # keep scoring isolated clients; readmit when evidence > -evidence_cut/2
    max_excluded: float = 1.0            # never leave out more than this fraction of clients in a round
    population: str = "kept"             # yardstick for z: "kept" clients or "all" not banned (isolated too)
    drill_down: bool = False             # Phase 4: flag parents whose mean update is out of line, then check
    group_cut: float = 2.0               # (parent z vs random same-size groups); members then face drill_cut
    drill_cut: float = 1.0
    drill_mode: str = "exclude"          # "exclude" (Phase 4 v1) | "evidence" (4b: member alignment -> evidence)
    min_group: int = 2                   # parents with fewer eligible members are not checked
    mad_floor: float = 0.0             # absolute cosine-score scale floor; 0 preserves previous runs
    relative_floor: float = 0.1
    isolation_patience: int = 1        # consecutive z < -evidence_cut, also requiring EMA below cut
    exclusion_patience: int = 1        # consecutive z < -z_cut before temporary exclusion
    two_sided: bool = False            # also act on z far above the others (colluders too close to the reference)
    gap_gate: float = 0.0              # > 0: bad evidence only in rounds whose scores split (max gap / median gap)
    global_rule: str = "sccf"          # Global step over kept clients: "sccf" (trust-weighted mean) | "arc_nnm_tm"
    global_f: int = 3                  # f for arc_nnm_tm before screening; reduced by the clients excluded
    global_f_fixed: bool = False       # keep f at global_f after exclusions (ALIE puzzle, candidate c)
    rewind: str = "none"               # H8: when a client is isolated, rewind the shared model to before its first
                                       # suspicious round: "none" | "trunk" | "trunk_root" (trunk and root head) |
                                       # "all" (every shared coordinate, personal heads included)
    rewind_window: int = 10            # Global models kept for the rewind (one per round)



@dataclass(frozen=True)
class RobustConfig:
    """Phase 5: an established rule in FedHEP's Global step; see rfl.server.robust_fedhep_server."""
    rule: str = "median"                # "median" | "trimmed_mean" | "bulyan" | "flame" | "fltrust" | "arc_nnm_tm"
    assumed_attackers: int = 3           # f for trimmed mean (trimmed per side) and Bulyan
    flame_lambda: float = 0.001          # FLAME's noise factor for image classification
    flame_on: str = "models"             # FLAME's cosine distances on "models" (paper's equations) or "updates"
    root_size: int = 100                 # FLTrust's root set (paper default)
    momentum: float = 0.0                # worker momentum beta over rounds, applied before the rule (0 = off)


@dataclass(frozen=True)
class BaselineConfig:                    # authors' METHODS params in src/baselines/experiment_configs.py
    fedprox_mu: float = 0.01
    ditto_lambda: float = 0.05
    krum_selected: int = 3               # m: updates Multi-Krum averages
    fedrep_head_epochs: int = 1


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "experiment"
    method: str = "fedavg"               # key in rfl.experiment.registry.METHODS
    model: str = "resnet9"
    data: DataConfig = field(default_factory=DataConfig)
    train: SGDSettings = field(default_factory=SGDSettings)
    run: RunConfig = field(default_factory=RunConfig)
    fedhep: FedHEPConfig = field(default_factory=FedHEPConfig)  # read by "fedhep" and "fedhep_defended"
    attack: AttackConfig = field(default_factory=AttackConfig)
    plus: PlusConfig = field(default_factory=PlusConfig)          # read only by "fedhep_plus"
    robust: RobustConfig = field(default_factory=RobustConfig)    # read only by "fedhep_robust"
    baseline: BaselineConfig = field(default_factory=BaselineConfig)


_SECTIONS: dict[str, type] = {"data": DataConfig, "train": SGDSettings, "run": RunConfig, "fedhep": FedHEPConfig,
                              "attack": AttackConfig, "baseline": BaselineConfig, "plus": PlusConfig,
                              "robust": RobustConfig}


def _build(cls: type, values: dict[str, Any], where: str) -> Any:
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown setting(s) in {where}: {sorted(unknown)}. Allowed: {sorted(known)}")
    kwargs = {
        k: _build(_SECTIONS[k], v or {}, f"{where}.{k}") if cls is ExperimentConfig and k in _SECTIONS else v
        for k, v in values.items()
    }
    defaults = {f.name: f.default for f in fields(cls)}
    for k, v in kwargs.items():
        if isinstance(defaults[k], int | float) and not isinstance(defaults[k], bool) and isinstance(v, str):
            raise ValueError(f"{where}.{k} must be a number, got the text {v!r} (YAML reads 1e9 as text; "
                             "write 1.0e+9)")
    return cls(**kwargs)


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    raw: dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text()) or {}
    for dotted, value in (overrides or {}).items():
        node = raw
        *parents, last = dotted.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[last] = value
    return _build(ExperimentConfig, raw, "config")


def to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    return dataclasses.asdict(cfg)
