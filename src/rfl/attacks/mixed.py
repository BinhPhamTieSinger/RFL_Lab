"""Mixed attacker populations (literature review Part 4, protocol M1-M6): several attacker subgroups in one run,
each with its own attack and its own first round. FedHEP-family servers only.

The run's attackers (sorted by id) are dealt to the parts in order: the first `count` to part 1, the next to
part 2, and so on. A part attacks from its `start_round` (staggered decoys, M3); before that its clients train
honestly. Coordinated, history and stealth parts see only their own subgroup as attackers, so colluder statistics
come from their own members, and the other subgroups' (already poisoned) uploads count among the "honest" updates
of an omniscient or stealth part. The backdoor success rate is measured with the first part that has a trigger."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from rfl.core.interfaces import Attack


class MixedAttack:
    def __init__(self, parts: Sequence[tuple[Attack, int, int]], attackers: frozenset[int]) -> None:
        if sum(n for _, n, _ in parts) != len(attackers):
            raise ValueError("the parts' counts must add up to the number of attackers")
        order = sorted(attackers)
        self.groups: list[tuple[Attack, frozenset[int], int]] = []
        for attack, n, start in parts:
            self.groups.append((attack, frozenset(order[:n]), start))
            order = order[n:]
        triggered = [a for a, _, _ in self.groups if hasattr(a, "trigger")]
        if triggered:
            self.trigger = triggered[0].trigger   # the runner measures backdoor success with it

    def part(self, client: int, round_idx: int) -> Attack | None:
        """The client's attack in this round, or None while its part has not started."""
        for attack, members, start in self.groups:
            if client in members:
                return attack if round_idx >= start else None
        return None

    def poison_batch(self, uploads: Mapping[int, torch.Tensor], start: torch.Tensor, round_idx: int,
                     parameter_mask: torch.Tensor) -> dict[int, torch.Tensor]:
        result = dict(uploads)
        for attack, members, first in self.groups:
            if hasattr(attack, "poison_batch") and round_idx >= first:
                result = attack.poison_batch(result, start, members, parameter_mask)
        return result
