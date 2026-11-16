"""FedHEP's client side, ported from the authors' `PyTorchLocalUpdater._update_hep` (src/core/updater.py) with the
settings their Table 2 runs use. Two parts:

* `FedHEPTrainer.train`: one round of local training of all three heads and the trunk at once, each head's loss
  weighted by how skewed the client's labels are (the "binomial" schedule). The authors' two-stage schedule
  (`high_cardinality_two_stage`) is off in every run they recorded, so it is not ported.
* `ensemble_logits`: how a client combines its three heads at test time.

What a client keeps between rounds is in `HeadMemory`. Parity with the authors' functions is tested in
tests/test_parity_fedhep.py."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from rfl.client.sgd import SGDSettings, cosine_lr
from rfl.core.types import ClientData
from rfl.models.resnet9 import MultiHeadResNet9

HEADS = ("local", "parent", "root")  # the order the authors use for weights and priors


@dataclass
class HeadMemory:
    """What one client carries from round to round."""

    # The authors store softmax probabilities here and feed them back as logits next round. Their starting value is
    # the ClientState default [0.333, 0.333, 0.334], not the heterogeneity prior. Both kept as in their code.
    ensemble_alpha: list[float] = field(default_factory=lambda: [0.333, 0.333, 0.334])
    local_head: dict[str, torch.Tensor] | None = None  # fc2_local after the last round; None = use the global one
    head_losses: dict[str, float] = field(default_factory=dict)  # last training batch's CE per head
    r_skew: float = 0.5
    active_mask: torch.Tensor | None = None  # bool [C]: classes in this client's training data


# --------------------------------------------------------------------------- label statistics and head weights

def label_stats(labels: torch.Tensor, num_classes: int) -> tuple[float, torch.Tensor, torch.Tensor]:
    """(r_skew, active-class mask, class counts). r_skew = (exp(H) - 1) / (C - 1), H the label entropy: 0 for a
    single class, 1 for uniform labels. Authors' `_compute_label_stats` + `compute_normalized_entropy_r_skew`."""
    device = labels.device
    if labels.numel() == 0:  # authors' fallback when a client has no data
        ones = torch.ones(num_classes, device=device)
        return 0.5, ones.bool(), ones
    active = torch.zeros(num_classes, dtype=torch.bool, device=device)
    counts = torch.zeros(num_classes, dtype=torch.float32, device=device)
    classes, n = torch.unique(labels, return_counts=True)
    active[classes.long()] = True
    counts[classes.long()] = n.float()
    if num_classes <= 1:
        return 1.0, active, counts
    p = counts / counts.sum()
    p = p[p > 0]
    entropy = -(p * torch.log(p + 1e-8)).sum().item()
    r = (math.exp(entropy) - 1.0) / float(num_classes - 1.0)
    return float(min(1.0, max(0.0, r))), active, counts


def binomial_head_weights(r_skew: float, num_classes: int, local_classes: int, num_groups: int
                          ) -> tuple[float, float, float]:
    """Loss weight per head (root, parent, local), summing to 1. Root: a + (1 - a) r^2, parent: 2 r (1 - r),
    local: (1 - r)^2, with floor a = max(1 / (2K), |Y_i| / C). Authors' `compute_dynamic_binomial_loss_weights`."""
    a = max(1.0 / (2.0 * max(1, num_groups)), float(local_classes) / float(max(1, num_classes)))
    lam_r = a + (1.0 - a) * r_skew ** 2
    lam_p = 2.0 * r_skew * (1.0 - r_skew)
    lam_l = (1.0 - r_skew) ** 2
    total = lam_r + lam_p + lam_l
    return lam_r / total, lam_p / total, lam_l / total


def heterogeneity_prior(r_skew: float, gamma: float = 1.0) -> tuple[float, float, float]:
    """Prior ensemble weight per head (local, parent, root): skewed clients lean local, uniform ones root."""
    pi_root = min(1.0, max(0.0, r_skew ** gamma))
    pi_local = (1.0 - pi_root) * (1.0 - r_skew)
    return pi_local, max(0.0, 1.0 - pi_root - pi_local), pi_root


# --------------------------------------------------------------------------- local training

class FedHEPTrainer:
    """`alpha_lr_scale` and `prior_mix` are the authors' `ensemble_alpha_lr_scale` and `prior_mix`. The local head
    sees the trunk's features with only `min(1, 10 / C)` of their gradient (their "dimensional capacity" anchor),
    so for CIFAR-100 90% of the local loss's gradient into the trunk is stopped."""

    def __init__(self, settings: SGDSettings, total_rounds: int, num_classes: int, num_groups: int,
                 alpha_lr_scale: float = 0.05, prior_mix: float = 0.5) -> None:
        self.s = settings
        self.total_rounds = total_rounds
        self.num_classes = num_classes
        self.num_groups = num_groups
        self.alpha_lr_scale = alpha_lr_scale
        self.prior_mix = prior_mix

    def train(self, model: MultiHeadResNet9, data: ClientData, round_idx: int, memory: HeadMemory,
              targets: torch.Tensor | None = None) -> None:
        """Trains `model` in place and updates `memory`. The caller has already loaded the global trunk and root
        head, the group's parent head, and (if any) this client's own local head. `targets` replaces the training
        labels (a label-flipping attacker); label statistics still come from the true labels, as in the
        authors' code."""
        C = self.num_classes
        device = data.x_train.device
        lr = cosine_lr(round_idx, self.total_rounds, self.s.lr, self.s.min_lr)
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=self.s.momentum, nesterov=self.s.nesterov,
                              weight_decay=self.s.weight_decay, foreach=False)

        r_skew, active, counts = label_stats(data.y_train, C)
        prior = torch.tensor(heterogeneity_prior(r_skew), dtype=torch.float32, device=device)
        alpha = torch.tensor(memory.ensemble_alpha, dtype=torch.float32, device=device)
        m_root, m_parent, m_local = binomial_head_weights(r_skew, C, int(active.sum().item()), self.num_groups)
        anchor = min(1.0, 10.0 / float(C))
        class_w = 1.0 / (counts ** 0.5 + 1e-4) * active.float()
        class_w = class_w / (class_w[class_w > 0].mean() + 1e-8)
        inactive = ~active.unsqueeze(0)
        last = {h: torch.zeros((), device=device) for h in HEADS}

        n = data.num_train
        batch = max(1, min(self.s.batch_size, n))
        model.train()
        for _ in range(self.s.local_epochs):
            order = torch.randperm(n, device=device)
            for i in range(0, n, batch):
                idx = order[i:i + batch]
                x, y = data.x_train[idx], (data.y_train if targets is None else targets)[idx]
                opt.zero_grad(set_to_none=True)
                if anchor < 1.0:
                    h = model.extract_features(x)
                    z_root, z_parent = model.fc2_root(h), model.fc2_parent(h)
                    z_local = model.fc2_local(anchor * h + (1.0 - anchor) * h.detach())
                else:
                    z_root, z_parent, z_local = model(x)

                loss = torch.zeros((), device=device)
                if m_root > 0:
                    ce = F.cross_entropy(z_root, y)
                    loss = loss + m_root * ce
                    last["root"] = ce.detach()
                if m_parent > 0:
                    ce = F.cross_entropy(z_parent.masked_fill(inactive, -1e9), y)
                    loss = loss + m_parent * ce
                    last["parent"] = ce.detach()
                if m_local > 0:
                    ce = F.cross_entropy(z_local.masked_fill(inactive, -1e9), y, weight=class_w)
                    loss = loss + m_local * ce
                    last["local"] = ce.detach()
                loss.backward()
                opt.step()
                alpha = self._alpha_step(alpha, prior, z_local, z_parent, z_root, y, lr)

        memory.ensemble_alpha = F.softmax(alpha, dim=0).cpu().tolist()
        memory.head_losses = {h: float(v) for h, v in last.items()}
        memory.r_skew = r_skew
        memory.active_mask = active
        memory.local_head = {k: v.detach().clone() for k, v in model.fc2_local.state_dict().items()}

    @torch.no_grad()
    def _alpha_step(self, alpha: torch.Tensor, prior: torch.Tensor, z_local: torch.Tensor, z_parent: torch.Tensor,
                    z_root: torch.Tensor, y: torch.Tensor, lr: float) -> torch.Tensor:
        """One gradient step on the ensemble mixing logits (cross-entropy of the blended logits), then a 50/50 mix
        with the prior. Uses the logits from before the optimizer step, as the authors do."""
        w = F.softmax(alpha, dim=0)
        p = F.softmax(w[0] * z_local + w[1] * z_parent + w[2] * z_root, dim=1)
        grad = (p - F.one_hot(y, p.shape[1]).to(p.dtype)) / y.shape[0]
        scale = torch.clamp(z_local.abs().mean(), min=1e-4)
        g = torch.stack([(grad * (z / scale)).sum() for z in (z_local, z_parent, z_root)])
        alpha = alpha - lr * self.alpha_lr_scale * g
        mixed = self.prior_mix * F.softmax(alpha, dim=0) + (1.0 - self.prior_mix) * prior
        return torch.log(mixed + 1e-8)


# --------------------------------------------------------------------------- test-time ensemble

def prediction_entropy(logits: torch.Tensor) -> float:
    p = F.softmax(logits, dim=1)
    return -(p * torch.log(p + 1e-8)).sum(dim=1).mean().item()


def confidence_head_weights(z_local: torch.Tensor, z_parent: torch.Tensor, z_root: torch.Tensor,
                            memory: HeadMemory, loss_beta: float = 1.0, temperature: float = 1.0
                            ) -> tuple[float, float, float]:
    """Per batch: score_h = -entropy_h - beta * last_train_loss_h + log(alpha_h); weights = softmax(score / T).
    Authors' `compute_head_weights` in "dynamic_confidence" mode."""
    scores = [-prediction_entropy(z) - loss_beta * memory.head_losses.get(h, 0.0)
              for z, h in zip((z_local, z_parent, z_root), HEADS, strict=True)]
    if len(memory.ensemble_alpha) == 3:
        scores = [s + math.log(max(1e-4, a)) for s, a in zip(scores, memory.ensemble_alpha, strict=True)]
    w = F.softmax(torch.tensor(scores, dtype=torch.float32) / temperature, dim=0)
    return w[0].item(), w[1].item(), w[2].item()


def ensemble_logits(model: MultiHeadResNet9, images: torch.Tensor, memory: HeadMemory,
                    iid_threshold: float = 0.85, fallback_temps: tuple[float, float, float] = (0.6, 0.8, 1.0)
                    ) -> torch.Tensor:
    """The client's prediction (authors' `evaluate_ensemble`, shared-backbone branch). Clients with near-uniform
    labels (r_skew >= 0.85) use the root head alone. Others blend the heads with `confidence_head_weights`, each
    head's logits rescaled to the root head's spread. Classes the client never trained on are pushed to -1e4."""
    z_root, z_parent, z_local = model(images)
    if memory.r_skew >= iid_threshold:
        w_l, w_p, w_r = 0.0, 0.0, 1.0
    else:
        w_l, w_p, w_r = confidence_head_weights(z_local, z_parent, z_root, memory)
    std_r = z_root.std().item()
    if std_r > 1e-6:
        t_l = max(0.2, min(2.0, z_local.std().item() / std_r))
        t_p = max(0.2, min(2.0, z_parent.std().item() / std_r))
        t_r = 1.0
    else:
        t_l, t_p, t_r = fallback_temps
    z = w_l * (z_local / t_l) + w_p * (z_parent / t_p) + w_r * (z_root / t_r)
    if memory.active_mask is not None:
        z = z.masked_fill(~memory.active_mask.to(z.device).unsqueeze(0), -1e4)
    return z


def head_agreement_logits(model: MultiHeadResNet9, images: torch.Tensor, memory: HeadMemory
                          ) -> tuple[torch.Tensor, int]:
    """Idea 9 (docs/IDEAS.md): the ensemble prediction, except where the client's two personal heads agree with
    each other and disagree with it; there the personal heads decide. A backdoor is learned by the shared trunk
    and mapped to its target by the shared root head, while the parent and local heads, trained on clean data,
    keep the true label (per-head success in docs/figures/phase6_backdoor_heads.png). Returns the logits and how
    many inputs were overruled. Evaluation only; training is unchanged."""
    z = ensemble_logits(model, images, memory)
    _, z_parent, z_local = model(images)
    if memory.active_mask is not None:
        keep = memory.active_mask.to(z.device).unsqueeze(0)
        z_parent = z_parent.masked_fill(~keep, -1e4)
        z_local = z_local.masked_fill(~keep, -1e4)
    p_parent, p_local, p_ens = z_parent.argmax(1), z_local.argmax(1), z.argmax(1)
    overrule = (p_parent == p_local) & (p_parent != p_ens)
    if not overrule.any():
        return z, 0
    personal = z_local / max(1e-6, z_local.std().item()) + z_parent / max(1e-6, z_parent.std().item())
    personal = personal * (z.std().item() / max(1e-6, personal.std().item()))
    return torch.where(overrule.unsqueeze(1), personal, z), int(overrule.sum().item())

