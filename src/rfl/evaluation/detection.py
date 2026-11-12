"""Detection quality per round, from a method's RoundReport and the true attackers.

Client level: precision, recall and F1 of the flagged set, and AUROC of the trust scores (the chance that a random
honest client is trusted more than a random attacker; 0.5 = no separation, ties count half). Parent level: a
parent is malicious when attackers are more than half of its members this round; precision/recall/F1 of the
flagged parents when the method checks parents. Undefined values (nothing flagged, no attackers, one class only)
are None."""
from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

from rfl.core.types import RoundReport


def prf(flagged: Collection[int], truth: Collection[int]) -> dict[str, float | None]:
    tp = len(set(flagged) & set(truth))
    precision = tp / len(flagged) if flagged else None
    recall = tp / len(truth) if truth else None
    if precision is None or recall is None:
        f1 = 0.0 if truth and tp == 0 else None
    else:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def auroc(trust: Mapping[int, float], attackers: Collection[int]) -> float | None:
    bad = [t for c, t in trust.items() if c in attackers]
    good = [t for c, t in trust.items() if c not in attackers]
    if not bad or not good:
        return None
    wins = sum(1.0 if g > b else 0.5 if g == b else 0.0 for g in good for b in bad)
    return wins / (len(good) * len(bad))


def malicious_groups(groups: Mapping[int, int], attackers: Collection[int]) -> set[int]:
    members: dict[int, list[int]] = {}
    for c, g in groups.items():
        members.setdefault(g, []).append(c)
    return {g for g, cs in members.items() if sum(c in attackers for c in cs) * 2 > len(cs)}


def detection_metrics(report: RoundReport, attackers: Collection[int],
                      num_clients: int | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"flagged": sorted(report.clients.flagged), **prf(report.clients.flagged, attackers),
                           "auroc": auroc(report.clients.trust, attackers)}
    if num_clients is not None:
        fp = len(set(report.clients.flagged) - set(attackers))
        honest_n = num_clients - len(set(attackers))
        out.update({"false_positives": fp, "true_positives": len(set(report.clients.flagged) & set(attackers)),
                    "false_exclusion_rate": fp / honest_n if honest_n else None})
    if report.groups is not None:
        bad_groups = malicious_groups(report.groups, attackers)
        out["malicious_groups"] = sorted(bad_groups)
        if report.flagged_groups is not None:
            out["flagged_groups"] = sorted(report.flagged_groups)
            out["group"] = prf(report.flagged_groups, bad_groups)
    return out
