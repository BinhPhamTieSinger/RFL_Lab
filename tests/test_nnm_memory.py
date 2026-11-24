"""nnm in blocks of rows (when the gathered tensor would not fit on the accelerator) gives the same result."""
import torch

from rfl.defense import classic


def test_blocked_nnm_equals_one_block(monkeypatch):
    torch.manual_seed(0)
    stack = torch.randn(50, 4000)
    full = classic.nnm(stack, 10)
    row_bytes = 40 * 4000 * 4
    for rows in (1, 7, 18, 49):   # forces blocks of these sizes, including uneven last blocks
        monkeypatch.setattr(classic, "_gather_budget", lambda device, r=rows: r * row_bytes)
        assert torch.equal(classic.nnm(stack, 10), full)


def test_no_budget_on_cpu():
    assert classic._gather_budget(torch.device("cpu")) is None
