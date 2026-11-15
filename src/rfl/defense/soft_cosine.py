"""SCCF: skew-calibrated soft-cosine aggregation, ported from the authors' `DeltaSpaceRobustAggregator` in
"soft_cosine" mode with its helpers `soft_cosine_trust`, `bound_update_norms` and `coordinate_median`
(src/core/aggregator.py). Used at both tiers of Defended FedHEP.

On flat updates (rfl.core.state):
 * BatchNorm buffers (running statistics, batch counters) are combined by coordinate-wise median;
 * weights: each update's norm is capped at k x the lower-quartile norm; each sender gets a trust weight
   softmax(cos(update, mean unit update) / T) and the result is the trust-weighted sum.
With class masks, the cosine only looks at the root head's weight rows for classes the sender holds ("skew
calibration"), so a client with few classes isn't penalised for touching few rows."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FlatLayout:
    """Where things sit in a flat state vector: which coordinates are buffers, and the root classifier weight's
    range counted among weight coordinates only. `params` lists each parameter tensor's (name, start, end) range,
    also in weight coordinates (used by layer-level screening)."""

    buffer_mask: torch.Tensor          # bool [D]
    classifier: tuple[int, int] | None
    params: tuple[tuple[str, int, int], ...] = ()


def flat_layout(state: Mapping[str, torch.Tensor], param_names: set[str]) -> FlatLayout:
    """Authors' `compute_buffer_slices` and `compute_classifier_slices`: the classifier is the first 2-D
    `.weight` whose name contains "fc2" (fc2_root for FedHEP's model)."""
    is_buffer: list[bool] = []
    classifier = None
    params: list[tuple[str, int, int]] = []
    n_weights = 0
    for k, v in state.items():
        if classifier is None and "fc2" in k and k.endswith(".weight") and v.dim() == 2:
            classifier = (n_weights, n_weights + v.numel())
        if k in param_names:
            params.append((k, n_weights, n_weights + v.numel()))
            n_weights += v.numel()
        is_buffer.extend([k not in param_names] * v.numel())
    return FlatLayout(torch.tensor(is_buffer), classifier, tuple(params))


def coordinate_median(stack: torch.Tensor) -> torch.Tensor:
    """Mean of the two middle values for an even count."""
    s = torch.sort(stack, dim=0).values
    n = s.shape[0]
    return s[n // 2] if n % 2 == 1 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def bound_update_norms(stack: torch.Tensor, k: float) -> torch.Tensor:
    """Scale each row down to at most k x the lower-quartile row norm."""
    if k <= 0:
        return stack
    norms = stack.norm(dim=1)
    lq = torch.sort(norms).values[min(norms.shape[0] - 1, int(0.25 * norms.shape[0]))]
    scale = torch.where(norms > 1e-8, (k * lq / norms).clamp(max=1.0), torch.ones_like(norms))
    return stack * scale.unsqueeze(1)


def soft_cosine_trust(updates: torch.Tensor, temperature: float, hard_rejection: bool, hard_threshold: float,
                      class_masks: torch.Tensor | None, classifier: tuple[int, int] | None) -> torch.Tensor:
    """[N] weights summing to 1. Everyone gets 1/N when the mean unit update is short (< 0.2), i.e. no
    consensus. With hard rejection, senders below `hard_threshold` cosine get zero weight. The temperature is
    tightened (x0.9) when cosines vary (variance > 0.05) and relaxed (x1.05) otherwise, each call afresh."""
    n = updates.shape[0]
    if n == 1:
        return torch.ones(1, device=updates.device)
    vecs, mask = updates, None
    if class_masks is not None and classifier is not None:
        s, e = classifier
        C = class_masks.shape[1]
        if e <= updates.shape[1] and (e - s) % C == 0:
            vecs = updates[:, s:e]
            mask = class_masks.float().to(updates.device).unsqueeze(2).expand(n, C, (e - s) // C).reshape(n, -1)
            vecs = vecs * mask
    unit = F.normalize(vecs, dim=1, eps=1e-10)
    centroid = unit.mean(dim=0)
    if centroid.norm() < 0.2:
        return torch.full((n,), 1.0 / n, device=updates.device)
    target = centroid.unsqueeze(0) * mask if mask is not None else centroid.unsqueeze(0).expand_as(unit)
    sims = F.cosine_similarity(unit, target, dim=1, eps=1e-10)
    if hard_rejection:
        sims = torch.where(sims < hard_threshold, torch.tensor(-1e9, device=updates.device), sims)
        if torch.all(sims <= -1e8):
            return torch.full((n,), 1.0 / n, device=updates.device)
    t = temperature
    valid = sims[sims > -1e8]
    if valid.shape[0] > 1:
        t = max(0.1, temperature * 0.9) if torch.var(valid).item() > 0.05 else min(2.0, temperature * 1.05)
    return F.softmax(sims / t, dim=0)


class SoftCosineAggregator:
    """Authors' settings: parents use k = 2 and no hard rejection; Global uses k = 3 and rejects cosine < -0.1.
    Both use T = 0.5 with the adaptive adjustment."""

    def __init__(self, layout: FlatLayout, norm_bound: float, hard_rejection: bool, hard_threshold: float = -0.1,
                 temperature: float = 0.5) -> None:
        self.layout = layout
        self.norm_bound = norm_bound
        self.hard_rejection = hard_rejection
        self.hard_threshold = hard_threshold
        self.temperature = temperature

    def aggregate(self, updates: Sequence[torch.Tensor], class_masks: torch.Tensor | None
                  ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (combined update, trust weight per sender). Senders must be finite (the server's sentinel
        already removed the others)."""
        stack = torch.stack(list(updates)).float()
        buf = self.layout.buffer_mask.to(stack.device)
        bounded = bound_update_norms(stack[:, ~buf], self.norm_bound)
        trust = soft_cosine_trust(bounded, self.temperature, self.hard_rejection, self.hard_threshold,
                                  class_masks, self.layout.classifier)
        out = torch.empty_like(stack[0])
        out[~buf] = (trust.unsqueeze(1) * bounded).sum(dim=0)
        out[buf] = coordinate_median(stack[:, buf])
        return out, trust
