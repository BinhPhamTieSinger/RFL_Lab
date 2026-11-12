"""FedRep (Collins et al., 2021), ported from the authors' `_update_split_head` in "fedrep" mode
(src/core/updater.py). Each client keeps a private classifier head (`fc2`). With a head from an earlier round it
trains `head_epochs` epochs head-only (body frozen, BatchNorm statistics paused, head gradient norm clipped to
1), then the remaining epochs jointly; in its first round all epochs are joint. It uploads the whole state with
the head coordinates set to zero, and is scored with its own trained model."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from rfl.client.sgd import LocalRule, SGDSettings, batches
from rfl.core.state import flatten
from rfl.core.types import ClientData

HEAD = "fc2"


def _set_bn(model: torch.nn.Module, train: bool) -> None:
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.train(train)


class FedRepTrainer(LocalRule):
    def __init__(self, settings: SGDSettings, total_rounds: int, head_epochs: int = 1) -> None:
        super().__init__(settings, total_rounds)
        self.head_epochs = head_epochs
        self.heads: dict[int, dict[str, torch.Tensor]] = {}
        self.personal: dict[int, torch.Tensor] = {}

    def train(self, model: torch.nn.Module, data: ClientData, round_idx: int, client_id: int,
              targets: torch.Tensor | None = None) -> None:
        if data.num_train == 0:
            return
        head: torch.nn.Module = getattr(model, HEAD)
        if client_id in self.heads:
            head.load_state_dict(self.heads[client_id])
            e_head = max(1, min(self.head_epochs, self.s.local_epochs))
            schedule = ["head"] * e_head + ["joint"] * (self.s.local_epochs - e_head)
        else:
            schedule = ["joint"] * self.s.local_epochs
        model.train()
        opt = self.sgd(model.parameters(), self.lr(round_idx))
        head_params = list(head.parameters())
        for phase in schedule:
            head_only = phase == "head"
            for name, p in model.named_parameters():
                p.requires_grad_(name.startswith(HEAD) or not head_only)
            _set_bn(model, not head_only)
            for x, y in batches(data, targets, 1, self.s.batch_size):
                opt.zero_grad(set_to_none=True)
                F.cross_entropy(model(x), y).backward()
                if head_only:
                    torch.nn.utils.clip_grad_norm_(head_params, max_norm=1.0)
                opt.step()
        model.train()
        for p in model.parameters():
            p.requires_grad_(True)
        self.heads[client_id] = {k: v.detach().clone() for k, v in head.state_dict().items()}
        self.personal[client_id] = flatten(model.state_dict())

    def upload(self, model: torch.nn.Module, client_id: int) -> torch.Tensor:
        """Whole state, head coordinates zeroed (the server never sees the head)."""
        vec = flatten(model.state_dict()).clone()
        offset = 0
        for k, v in model.state_dict().items():
            if k.startswith(HEAD + "."):
                vec[offset:offset + v.numel()] = 0.0
            offset += v.numel()
        return vec

    def personal_model(self, client_id: int) -> torch.Tensor | None:
        return self.personal.get(client_id)
