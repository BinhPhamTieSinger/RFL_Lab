"""Temporal trust tracking, ported from `UnifiedDefendedEngine.run_round` (src/core/unified_defended_engine.py).

Each time a sender gets a trust weight t among n senders, its score moves toward min(1, t * n) (1 = an equal
share) with momentum 0.85. A score below 0.25 isolates the client for the rest of training. Scores start at 1
and are updated once per tier per round (parent tier, then Global)."""
from __future__ import annotations

from collections.abc import Mapping


class TrustLedger:
    def __init__(self, num_clients: int, momentum: float = 0.85, threshold: float = 0.25) -> None:
        self.score = {c: 1.0 for c in range(num_clients)}
        self.isolated: set[int] = set()
        self.momentum = momentum
        self.threshold = threshold

    def update(self, trust: Mapping[int, float], num_senders: int) -> None:
        for c, t in trust.items():
            self.score[c] = self.momentum * self.score[c] + (1.0 - self.momentum) * min(1.0, t * num_senders)
            if self.score[c] < self.threshold:
                self.isolated.add(c)
