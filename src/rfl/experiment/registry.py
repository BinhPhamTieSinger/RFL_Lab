"""Name -> components. Adding a method means adding one entry here; nothing else changes."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

import torch

from rfl.aggregation.fedavg import FedAvg
from rfl.aggregation.multikrum import MultiKrum
from rfl.attacks.adaptive import CoordinatedAttack, HistoryAttack, StealthBackdoorAttack
from rfl.attacks.backdoor import BackdoorAttack
from rfl.attacks.byzantine import ByzantineAttack, select_attackers, select_attackers_fixed
from rfl.attacks.mixed import MixedAttack
from rfl.client.ditto import DittoTrainer
from rfl.client.fedhep import FedHEPTrainer
from rfl.client.fedprox import FedProxTrainer
from rfl.client.fedrep import FedRepTrainer
from rfl.client.scaffold import ScaffoldTrainer
from rfl.client.sgd import LocalRule, SGDTrainer
from rfl.core.interfaces import Aggregator, Attack, FederatedServer
from rfl.core.types import ClientData
from rfl.defense.screening import ScreenOptions
from rfl.experiment.config import AttackConfig, ExperimentConfig
from rfl.grouping.similarity import SimilarityGrouping, concentrate, spread
from rfl.models.resnet9 import MultiHeadResNet9, ResNet9
from rfl.server.defended_fedhep_server import DefendedFedHEPServer
from rfl.server.fedavg_server import ConsensusServer
from rfl.server.fedhep_plus_server import FedHEPPlusServer
from rfl.server.fedhep_server import FedHEPServer
from rfl.server.robust_fedhep_server import RobustFedHEPServer

MODELS: dict[str, Callable[[int, int], torch.nn.Module]] = {
    "resnet9": lambda in_channels, num_classes: ResNet9(in_channels, num_classes),
    "resnet9_multihead": lambda in_channels, num_classes: MultiHeadResNet9(in_channels, num_classes),
}


def _single_model(cfg: ExperimentConfig, model: torch.nn.Module, clients: Sequence[ClientData], rule: LocalRule,
                  aggregator: Aggregator | None = None, personalised: bool = False) -> FederatedServer:
    attack, attackers = _attack(cfg, len(clients))
    num_classes = int(model.fc2.out_features) if isinstance(model, ResNet9) else 0
    return ConsensusServer(model, clients, rule, aggregator or FedAvg(), attack=attack, attackers=attackers,
                           personalised=personalised, num_classes=num_classes)


def build_fedavg(cfg: ExperimentConfig, model: torch.nn.Module, clients: Sequence[ClientData]) -> FederatedServer:
    return _single_model(cfg, model, clients, SGDTrainer(cfg.train, cfg.run.rounds))


def build_fedprox(cfg: ExperimentConfig, model: torch.nn.Module, clients: Sequence[ClientData]) -> FederatedServer:
    return _single_model(cfg, model, clients, FedProxTrainer(cfg.train, cfg.run.rounds, cfg.baseline.fedprox_mu))


def build_scaffold(cfg: ExperimentConfig, model: torch.nn.Module, clients: Sequence[ClientData]
                   ) -> FederatedServer:
    return _single_model(cfg, model, clients, ScaffoldTrainer(cfg.train, cfg.run.rounds))


def build_multikrum(cfg: ExperimentConfig, model: torch.nn.Module, clients: Sequence[ClientData]
                    ) -> FederatedServer:
    _, attackers = _attack(cfg, len(clients))
    f = max(1, len(attackers)) if attackers else None  # the authors' oracle: f = true number of attackers
    return _single_model(cfg, model, clients, SGDTrainer(cfg.train, cfg.run.rounds),
                         MultiKrum(cfg.baseline.krum_selected, f))


def build_ditto(cfg: ExperimentConfig, model: torch.nn.Module, clients: Sequence[ClientData]) -> FederatedServer:
    return _single_model(cfg, model, clients, DittoTrainer(cfg.train, cfg.run.rounds, cfg.baseline.ditto_lambda),
                         personalised=True)


def build_fedrep(cfg: ExperimentConfig, model: torch.nn.Module, clients: Sequence[ClientData]) -> FederatedServer:
    rule = FedRepTrainer(cfg.train, cfg.run.rounds, cfg.baseline.fedrep_head_epochs)
    return _single_model(cfg, model, clients, rule, personalised=True)


def _attack(cfg: ExperimentConfig, num_clients: int) -> tuple[Attack | None, frozenset[int]]:
    a = cfg.attack
    if a.kind in ("alie", "min_max", "ipm", "min_sum", "fang_trim", "poisonedfl", "xfed",
                  "stealth_backdoor", "mixed") and cfg.method not in (
            "fedhep", "fedhep_defended", "fedhep_plus", "fedhep_robust"):
        raise ValueError("Coordinated attacks currently require a FedHEP server")
    if a.kind is None:
        return None, frozenset()
    if a.kind == "mixed":
        if not a.mix or a.count is not None:
            raise ValueError("attack.kind mixed takes its attackers from attack.mix (leave attack.count unset)")
        parts = [(replace(a, **{k: v for k, v in p.items() if k not in ("count", "start_round")}),
                  int(p["count"]), int(p.get("start_round", 0))) for p in a.mix]
        if any(p.kind in ("mixed", None) for p, _, _ in parts):
            raise ValueError("each part of attack.mix needs its own kind (not mixed)")
        attackers = select_attackers_fixed(num_clients, sum(n for _, n, _ in parts), cfg.run.seed)
        return MixedAttack([(_one_attack(p, cfg.run.seed), n, s) for p, n, s in parts], attackers), attackers
    attackers = (select_attackers_fixed(num_clients, a.count, cfg.run.seed) if a.count is not None
                 else select_attackers(num_clients, a.rate, cfg.run.seed))
    return _one_attack(a, cfg.run.seed), attackers


def _one_attack(a: AttackConfig, seed: int) -> Attack:
    stealth = a.kind == "stealth_backdoor"
    return (BackdoorAttack(a.backdoor_target, a.backdoor_fraction, a.scale, seed) if a.kind == "backdoor" else
            StealthBackdoorAttack(a.backdoor_target, a.backdoor_fraction, seed) if stealth else
            CoordinatedAttack(a.kind, a.ipm_epsilon if a.kind == "ipm" else a.alie_z, a.knowledge,
                              a.perturbation, seed)
            if a.kind in ("alie", "min_max", "ipm", "min_sum", "fang_trim") else
            HistoryAttack(a.kind, seed) if a.kind in ("poisonedfl", "xfed") else
            ByzantineAttack(a.kind))


def _fedhep_parts(cfg: ExperimentConfig, model: torch.nn.Module, clients: Sequence[ClientData]
                  ) -> tuple[MultiHeadResNet9, FedHEPTrainer, SimilarityGrouping]:
    if not isinstance(model, MultiHeadResNet9):
        raise ValueError(f"method {cfg.method!r} needs model 'resnet9_multihead'")
    groups = max(1, min(cfg.fedhep.num_groups, len(clients)))  # the authors clamp the same way
    trainer = FedHEPTrainer(cfg.train, cfg.run.rounds, model.fc2_root.out_features, groups)
    grouping = SimilarityGrouping(len(clients), groups, cfg.run.seed)
    attackers = _attack(cfg, len(clients))[1]
    if cfg.attack.placement == "concentrated":
        grouping.assignment = concentrate(grouping.assignment, attackers, groups)
    elif cfg.attack.placement == "spread":
        grouping.assignment = spread(grouping.assignment, attackers, groups)
    elif cfg.attack.placement != "random":
        raise ValueError(f"Unknown attack placement {cfg.attack.placement!r}")
    return model, trainer, grouping


def build_fedhep(cfg: ExperimentConfig, model: torch.nn.Module, clients: Sequence[ClientData]) -> FederatedServer:
    attack, attackers = _attack(cfg, len(clients))
    hep, trainer, grouping = _fedhep_parts(cfg, model, clients)
    return FedHEPServer(hep, clients, trainer, grouping, attack=attack, attackers=attackers,
                        similarity_on=cfg.fedhep.similarity_on)


def build_fedhep_defended(cfg: ExperimentConfig, model: torch.nn.Module, clients: Sequence[ClientData]
                          ) -> FederatedServer:
    attack, attackers = _attack(cfg, len(clients))
    hep, trainer, grouping = _fedhep_parts(cfg, model, clients)
    return DefendedFedHEPServer(hep, clients, trainer, grouping, attack=attack, attackers=attackers)


def build_fedhep_plus(cfg: ExperimentConfig, model: torch.nn.Module, clients: Sequence[ClientData]
                      ) -> FederatedServer:
    attack, attackers = _attack(cfg, len(clients))
    hep, trainer, grouping = _fedhep_parts(cfg, model, clients)
    p = cfg.plus
    screen = ScreenOptions(p.signal, p.reference, p.z_cut, p.evidence_cut, p.evidence_decay,
                           readmit=p.readmit, max_excluded=p.max_excluded, population=p.population,
                           drill_down=p.drill_down, group_cut=p.group_cut, drill_cut=p.drill_cut,
                           drill_mode=p.drill_mode, min_group=p.min_group, mad_floor=p.mad_floor,
                           relative_floor=p.relative_floor, isolation_patience=p.isolation_patience,
                           exclusion_patience=p.exclusion_patience, two_sided=p.two_sided, gap_gate=p.gap_gate,
                           critical_k=p.critical_k, sign_topk=p.sign_topk,
                           peer_share=p.peer_share, shadow_reference=p.shadow_reference,
                           high_gate=p.high_gate, sentence=p.sentence, max_strikes=p.max_strikes,
                           probation_patience=p.probation_patience, sentence_single=p.sentence_single,
                           isolation_window=p.isolation_window)
    probe = None
    if p.signal in ("probe_logits", "head_gap"):
        probe = _probe_set(cfg, clients, p.probe_size)
    return FedHEPPlusServer(hep, clients, trainer, grouping, screen, regroup=p.regroup, attack=attack,
                            attackers=attackers, global_rule=p.global_rule, global_f=p.global_f, probe=probe,
                            global_f_fixed=p.global_f_fixed, rewind=p.rewind, rewind_window=p.rewind_window,
                            rewind_agree=p.rewind_agree, global_f_floor=p.global_f_floor)


def _probe_set(cfg: ExperimentConfig, clients: Sequence[ClientData], size: int) -> torch.Tensor:
    """Probe images for output-space screening: the FLTrust root-set draw (outside every client's data), images
    only."""
    d = cfg.data
    device = clients[0].x_train.device
    if d.dataset == "cifar100":
        from rfl.data.cifar100 import load_root_set
        x, _ = load_root_set(d.root, d.train_subset, d.test_subset, cfg.run.seed, size, device)
        return x
    from rfl.experiment.runner import synthetic_data
    _, _, xte, _ = synthetic_data(d.train_subset or 600, (d.test_subset or 200) + size, 10, cfg.run.seed, device)
    return xte[-size:]


def _root_set(cfg: ExperimentConfig, clients: Sequence[ClientData]) -> ClientData:
    """FLTrust's root set, repeated up to the median client's training-set size so the server takes as many
    local SGD steps as a typical client (the paper's server and clients run the same number of iterations)."""
    d, size = cfg.data, cfg.robust.root_size
    device = clients[0].x_train.device
    if d.dataset == "cifar100":
        from rfl.data.cifar100 import load_root_set
        x, y = load_root_set(d.root, d.train_subset, d.test_subset, cfg.run.seed, size, device)
    else:  # synthetic: extra draws after the test set, from the same class patterns
        from rfl.experiment.runner import synthetic_data
        _, _, xte, yte = synthetic_data(d.train_subset or 600, (d.test_subset or 200) + size, 10, cfg.run.seed,
                                        device)
        x, y = xte[-size:], yte[-size:]
    target = sorted(c.num_train for c in clients)[len(clients) // 2]
    idx = torch.arange(max(target, size), device=device) % size
    return ClientData(x[idx], y[idx], x[:0], y[:0])


def build_fedhep_robust(cfg: ExperimentConfig, model: torch.nn.Module, clients: Sequence[ClientData]
                        ) -> FederatedServer:
    attack, attackers = _attack(cfg, len(clients))
    hep, trainer, grouping = _fedhep_parts(cfg, model, clients)
    r = cfg.robust
    root = _root_set(cfg, clients) if r.rule == "fltrust" else None
    return RobustFedHEPServer(hep, clients, trainer, grouping, r.rule, r.assumed_attackers, r.flame_lambda,
                              r.flame_on, root, cfg.run.seed, attack=attack, attackers=attackers,
                              momentum=r.momentum)


METHODS: dict[str, Callable[[ExperimentConfig, torch.nn.Module, Sequence[ClientData]], FederatedServer]] = {
    "fedavg": build_fedavg,
    "fedprox": build_fedprox,
    "scaffold": build_scaffold,
    "multikrum": build_multikrum,
    "ditto": build_ditto,
    "fedrep": build_fedrep,
    "fedhep": build_fedhep,
    "fedhep_defended": build_fedhep_defended,
    "fedhep_plus": build_fedhep_plus,
    "fedhep_robust": build_fedhep_robust,
}
