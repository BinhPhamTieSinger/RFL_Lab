"""A model's full state as one flat float vector, the representation the authors' server works on
(`model_to_vector` / `vector_to_model` in their src/core/model.py).

The vector holds every state_dict entry in state_dict order, including BatchNorm running statistics and
BatchNorm's integer batch counter cast to float. Order matters: the random projection used for grouping maps
coordinates by position, so a different order would give different groups."""
from __future__ import annotations

from collections.abc import Mapping

import torch


def flatten(state: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.detach().reshape(-1).float() for v in state.values()])


def unflatten(vector: torch.Tensor, like: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Inverse of `flatten`. Each entry gets the dtype of `like` (float to int64 truncates, as `copy_` does in
    the authors' `vector_to_model`)."""
    out: dict[str, torch.Tensor] = {}
    offset = 0
    for k, v in like.items():
        n = v.numel()
        out[k] = vector[offset:offset + n].reshape(v.shape).to(v.dtype)
        offset += n
    if offset != vector.numel():
        raise ValueError(f"vector has {vector.numel()} entries, state needs {offset}")
    return out
