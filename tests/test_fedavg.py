"""Property-based tests: each test states an invariant FedAvg must satisfy for *any* input Hypothesis generates."""
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from rfl.aggregation.fedavg import FedAvg
from rfl.core.types import ClientUpdate

finite = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)


@st.composite
def updates(draw, min_clients=1, max_clients=8):
    n = draw(st.integers(min_clients, max_clients))
    size = draw(st.integers(1, 6))
    return [
        ClientUpdate(
            client_id=i,
            num_samples=draw(st.integers(1, 500)),
            delta={"w": torch.tensor(draw(st.lists(finite, min_size=size, max_size=size)), dtype=torch.float64)},
        )
        for i in range(n)
    ]


@settings(max_examples=200)
@given(updates())
def test_result_stays_inside_the_inputs(us):
    """A weighted mean never leaves the per-coordinate range of what it averages."""
    out = FedAvg().aggregate(us)["w"]
    stack = torch.stack([u.delta["w"] for u in us])
    assert torch.all(out >= stack.min(0).values - 1e-9) and torch.all(out <= stack.max(0).values + 1e-9)


@given(updates(), st.randoms())
def test_order_of_clients_does_not_matter(us, rnd):
    shuffled = list(us)
    rnd.shuffle(shuffled)
    assert torch.allclose(FedAvg().aggregate(us)["w"], FedAvg().aggregate(shuffled)["w"])


@given(updates())
def test_identical_updates_return_that_update(us):
    same = [ClientUpdate(u.client_id, u.num_samples, us[0].delta) for u in us]
    assert torch.allclose(FedAvg().aggregate(same)["w"], us[0].delta["w"])


@given(updates(min_clients=2))
def test_explicit_equal_weights_give_plain_mean(us):
    out = FedAvg().aggregate(us, weights={u.client_id: 1.0 for u in us})["w"]
    assert torch.allclose(out, torch.stack([u.delta["w"] for u in us]).mean(0))


@given(updates(min_clients=2))
def test_zero_weight_removes_a_client(us):
    """Trust weight 0 must exclude a sender completely: the contract detectors rely on."""
    weights = {u.client_id: 1.0 for u in us}
    weights[us[0].client_id] = 0.0
    out = FedAvg().aggregate(us, weights=weights)["w"]
    rest = torch.stack([u.delta["w"] for u in us[1:]]).mean(0)
    assert torch.allclose(out, rest)


def test_rejects_empty_and_all_zero_weights():
    u = ClientUpdate(0, 10, {"w": torch.ones(2)})
    with pytest.raises(ValueError):
        FedAvg().aggregate([])
    with pytest.raises(ValueError):
        FedAvg().aggregate([u], weights={0: 0.0})
