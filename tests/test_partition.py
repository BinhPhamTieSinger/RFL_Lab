"""Invariants every partitioner must satisfy, checked on generated label arrays."""
import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from rfl.data.partition import DirichletPartitioner, IIDPartitioner

labels = st.integers(20, 400).flatmap(
    lambda n: st.lists(st.integers(0, 9), min_size=n, max_size=n).map(np.array))


@given(labels, st.integers(2, 12), st.sampled_from([0.05, 0.1, 0.5, 1.0, 10.0]), st.integers(0, 10_000))
def test_dirichlet_assigns_every_sample_exactly_once(y, n_clients, alpha, seed):
    parts = DirichletPartitioner(alpha).split(y, n_clients, seed)
    assert len(parts) == n_clients
    allidx = np.concatenate(parts)
    assert sorted(allidx.tolist()) == list(range(len(y)))


@given(labels, st.integers(2, 12), st.integers(0, 10_000))
def test_iid_gives_equal_disjoint_shares(y, n_clients, seed):
    parts = IIDPartitioner().split(y, n_clients, seed)
    sizes = {len(p) for p in parts}
    assert sizes == {len(y) // n_clients}
    allidx = np.concatenate(parts)
    assert len(set(allidx.tolist())) == len(allidx)


@given(labels, st.integers(0, 10_000))
def test_same_seed_same_split(y, seed):
    a = DirichletPartitioner(0.3).split(y, 5, seed)
    b = DirichletPartitioner(0.3).split(y, 5, seed)
    assert all(np.array_equal(x, z) for x, z in zip(a, b, strict=True))
