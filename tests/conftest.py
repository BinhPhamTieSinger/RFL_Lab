"""Hypothesis settings shared by all tests. No per-example deadline: the first PyTorch call warms up slowly."""
from hypothesis import settings

settings.register_profile("rfl", deadline=None, max_examples=100)
settings.load_profile("rfl")
