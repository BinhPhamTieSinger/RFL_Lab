"""Output-space screening signals (Phase 6, 2026-11-15): what a client's uploaded model *does* on a small server
probe set, instead of how its weights point. Motivation (EXPERIMENT_LOG, 2026-11-15): the backdoor lives in the
shared trunk and its attackers never split off in weight-cosine space, but a backdoored model behaves differently.

Two signals per uploaded model, computed on the probe images only (no labels needed):
  * "probe_logits": the root head's logits, flattened over the probe set, as the screening vector; the screen then
    takes cosines to the reference as for weights, so an upload whose predictions diverge from the consensus scores
    low;
  * "head_gap": mean KL(root || local) + KL(root || parent) over the probe set, a scalar. A backdoor sits in the
    trunk, so all three heads shift together (small gap); honest personalization moves the local head only (larger
    gap). Which direction is anomalous is an empirical question, so the screen's two-sided option applies.
The probe set is the FLTrust root set (rfl.data.cifar100.load_root_set): images outside every client's data."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from rfl.core.state import unflatten
from rfl.models.resnet9 import MultiHeadResNet9


@torch.no_grad()
def probe_outputs(model: MultiHeadResNet9, template: dict[str, torch.Tensor], upload: torch.Tensor,
                  images: torch.Tensor, chunk: int = 512) -> tuple[torch.Tensor, float]:
    """(root-head logits [P, C] flattened to [P*C], head gap) of the uploaded flat state on the probe images."""
    model.load_state_dict(unflatten(upload, template))
    model.eval()
    roots, gaps = [], []
    for i in range(0, images.shape[0], chunk):
        z_root, z_parent, z_local = model(images[i:i + chunk])
        p_root = F.log_softmax(z_root, dim=1)
        kl_local = F.kl_div(F.log_softmax(z_local, dim=1), p_root, log_target=True, reduction="batchmean")
        kl_parent = F.kl_div(F.log_softmax(z_parent, dim=1), p_root, log_target=True, reduction="batchmean")
        roots.append(z_root)
        gaps.append(float(kl_local + kl_parent) * z_root.shape[0])
    return torch.cat(roots).reshape(-1), sum(gaps) / images.shape[0]
