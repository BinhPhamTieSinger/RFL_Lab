"""Inference-time backdoor guard from exits at several depths (Phase 6, 2026-11-16), after Orion (Huang et al.,
IJCAI 2023), which attaches classifiers to intermediate layers of a single trained model and flags inputs whose
deep prediction disagrees with the shallow ones. Not in any federated method we know of.

Why for FedHEP: a backdoor implanted in the shared trunk survives after its attackers are isolated and keeps
growing (EXPERIMENT_LOG, 2026-11-15). Detection-then-isolation cannot remove it; a guard on each honest client's
predictions can still refuse it. A trigger has to pass through the deep layers to change the class, so classifiers
reading shallower layers, trained on the client's own clean data, tend to keep the true label.

Per honest client, at each evaluation:
  1. load the client's trained model (frozen); compute global-average-pooled features after `res1` (64 channels),
     `layer2` (128) and `layer3` (256) on its training images;
  2. split them 80/20 with a generator seeded by the client id (no global randomness is used);
  3. fit one linear classifier per depth on the 80% by full-batch gradient descent on the cross-entropy over the
     classes the client holds (weights start at zero, so the fit is deterministic);
  4. score s(x) = 1 - mean over exits of p_exit(final class | x); threshold tau = the `quantile` of s on the 20%;
  5. predict: the model's own ensemble prediction, except where s(x) > tau, where the exits' averaged probabilities
     decide.
Training is untouched: the guard only reads trained models, and its fitting happens in evaluation rounds."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from rfl.models.resnet9 import MultiHeadResNet9


def depth_features(model: MultiHeadResNet9, x: torch.Tensor) -> list[torch.Tensor]:
    """Pooled features after res1, layer2 and layer3 (the trunk's forward pass, stopped early)."""
    out = model.layer1(model.prep(x))
    out = model.res1(out) + out
    f1 = out.mean(dim=(2, 3))
    out = model.layer2(out)
    f2 = out.mean(dim=(2, 3))
    out = model.layer3(out)
    f3 = out.mean(dim=(2, 3))
    return [f1, f2, f3]


class _Exit:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> None:
        self.mean, self.std, self.weight, self.bias = mean, std, weight, bias

    def logits(self, feats: torch.Tensor) -> torch.Tensor:
        return ((feats - self.mean) / self.std) @ self.weight.T + self.bias


class ExitGuard:
    def __init__(self, num_classes: int, quantile: float = 0.95, steps: int = 200, lr: float = 0.5,
                 holdout: float = 0.2, seed: int = 0, chunk: int = 1024) -> None:
        self.num_classes, self.quantile, self.steps, self.lr = num_classes, quantile, steps, lr
        self.holdout, self.seed, self.chunk = holdout, seed, chunk
        self.exits: dict[int, list[_Exit]] = {}
        self.tau: dict[int, float] = {}
        self.flagged = 0
        self.seen = 0

    @torch.no_grad()
    def _features(self, server: Any, cid: int, images: torch.Tensor) -> list[torch.Tensor]:
        server._load(server.trained[cid])
        server.model.eval()
        parts = [depth_features(server.model, images[i:i + self.chunk])
                 for i in range(0, images.shape[0], self.chunk)]
        return [torch.cat([p[d] for p in parts]) for d in range(3)]

    def _active(self, server: Any, cid: int, device: torch.device) -> torch.Tensor:
        mask = server.memory[cid].active_mask
        return (torch.ones(self.num_classes, dtype=torch.bool) if mask is None else mask).to(device)

    def fit(self, server: Any, clients: Any, exclude: frozenset[int]) -> None:
        """Fit every honest client's exits and threshold on its own training data."""
        self.flagged = self.seen = 0
        for cid, data in enumerate(clients):
            if cid in exclude or data.num_train < 10:
                continue
            feats = self._features(server, cid, data.x_train)
            y = data.y_train
            active = self._active(server, cid, y.device)
            g = torch.Generator().manual_seed(self.seed * 1_000_003 + cid)
            order = torch.randperm(y.shape[0], generator=g).to(y.device)
            n_cal = max(1, int(self.holdout * y.shape[0]))
            cal, fit = order[:n_cal], order[n_cal:]
            exits = [self._fit_exit(f[fit], y[fit], active) for f in feats]
            self.exits[cid] = exits
            with torch.no_grad():
                final = self._final_class(server, cid, data.x_train[cal])
                self.tau[cid] = float(torch.quantile(self._score([f[cal] for f in feats], exits, final, active),
                                                     self.quantile))

    def _fit_exit(self, feats: torch.Tensor, y: torch.Tensor, active: torch.Tensor) -> _Exit:
        mean = feats.mean(dim=0)
        std = feats.std(dim=0).clamp_min(1e-6)
        z = (feats - mean) / std
        weight = torch.zeros(self.num_classes, z.shape[1], device=z.device, requires_grad=True)
        bias = torch.zeros(self.num_classes, device=z.device, requires_grad=True)
        with torch.enable_grad():
            for _ in range(self.steps):
                logits = (z @ weight.T + bias).masked_fill(~active, -1e9)
                loss = F.cross_entropy(logits, y)
                gw, gb = torch.autograd.grad(loss, (weight, bias))
                with torch.no_grad():
                    weight -= self.lr * gw
                    bias -= self.lr * gb
        return _Exit(mean, std, weight.detach(), bias.detach())

    def _probs(self, feats: list[torch.Tensor], exits: list[_Exit], active: torch.Tensor) -> torch.Tensor:
        return torch.stack([F.softmax(e.logits(f).masked_fill(~active, -1e9), dim=1)
                            for f, e in zip(feats, exits, strict=True)]).mean(dim=0)

    def _score(self, feats: list[torch.Tensor], exits: list[_Exit], final: torch.Tensor,
               active: torch.Tensor) -> torch.Tensor:
        p = self._probs(feats, exits, active)
        return 1.0 - p.gather(1, final.unsqueeze(1)).squeeze(1)

    @torch.no_grad()
    def _final_class(self, server: Any, cid: int, images: torch.Tensor) -> torch.Tensor:
        return torch.cat([server.predict(cid, images[i:i + self.chunk]).argmax(dim=1)
                          for i in range(0, images.shape[0], self.chunk)])

    @torch.no_grad()
    def predict(self, server: Any, cid: int, images: torch.Tensor) -> torch.Tensor:
        """The client's guarded logits. Clients without fitted exits (attackers) get the plain prediction."""
        logits = server.predict(cid, images)
        if cid not in self.exits:
            return logits
        active = self._active(server, cid, images.device)
        feats = self._features(server, cid, images)
        final = logits.argmax(dim=1)
        score = self._score(feats, self.exits[cid], final, active)
        swap = score > self.tau[cid]
        self.flagged += int(swap.sum())
        self.seen += int(swap.numel())
        if swap.any():
            p = self._probs([f[swap] for f in feats], self.exits[cid], active)
            logits = logits.clone()
            logits[swap] = torch.log(p.clamp_min(1e-12)).masked_fill(~active, -1e4).to(logits.dtype)
        return logits
