"""SCAFFOLD (Karimireddy et al., 2020), ported from the authors' `ScaffoldUpdater.update_scaffold` and
`ScaffoldEngine` (src/baselines/): local SGD without momentum (weight decay 1e-4 kept) whose gradients are
corrected by c - c_i; afterwards c_i += delta_c with delta_c = -c + (w_start - w_end) / (K * lr) ("option II"),
and the server adds the mean delta_c of all clients to c. Control variates cover the parameters only."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from rfl.client.sgd import LocalRule, SGDSettings, batches
from rfl.core.types import ClientData


class ScaffoldTrainer(LocalRule):
    def __init__(self, settings: SGDSettings, total_rounds: int) -> None:
        super().__init__(settings, total_rounds)
        self.server_c: torch.Tensor | None = None
        self.client_c: dict[int, torch.Tensor] = {}
        self._round_deltas: list[torch.Tensor] = []

    def train(self, model: torch.nn.Module, data: ClientData, round_idx: int, client_id: int,
              targets: torch.Tensor | None = None) -> None:
        if data.num_train == 0:
            return
        params = list(model.parameters())
        w_start = torch.nn.utils.parameters_to_vector(params).detach().clone()
        if self.server_c is None:
            self.server_c = torch.zeros_like(w_start)
        c_i = self.client_c.get(client_id, torch.zeros_like(w_start))
        correction = self.server_c - c_i
        lr = self.lr(round_idx)
        opt = torch.optim.SGD(params, lr=lr, momentum=0.0, weight_decay=self.s.weight_decay, foreach=False)
        model.train()
        steps = 0
        for x, y in batches(data, targets, self.s.local_epochs, self.s.batch_size):
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(x), y).backward()
            offset = 0
            for p in params:
                if p.grad is not None:
                    p.grad.data.add_(correction[offset:offset + p.numel()].view(p.shape))
                offset += p.numel()
            opt.step()
            steps += 1
        w_end = torch.nn.utils.parameters_to_vector(params).detach()
        step_size = max(1, steps) * lr  # (1 / s) * x as the authors write it: x / s rounds differently
        delta_c = -self.server_c + (1.0 / step_size) * (w_start - w_end) if step_size > 1e-8 else -self.server_c
        self.client_c[client_id] = c_i + delta_c
        self._round_deltas.append(delta_c)

    def end_round(self) -> None:
        if self._round_deltas and self.server_c is not None:
            self.server_c = self.server_c + torch.stack(self._round_deltas).mean(dim=0)
        self._round_deltas = []
