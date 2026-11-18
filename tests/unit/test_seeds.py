"""Tests for :func:`day0.seeds.derive_seed`."""

from __future__ import annotations

from day0.seeds import derive_seed


def test_same_inputs_give_same_seed() -> None:
    """The derivation is a pure function of its inputs."""
    assert derive_seed(1234, "perf", 8) == derive_seed(1234, "perf", 8)


def test_result_is_in_31_bit_range() -> None:
    """Seeds fit in a signed 32-bit integer, so every consumer accepts them."""
    for base in range(0, 200):
        seed = derive_seed(base, "label", base)
        assert 0 <= seed <= 2**31 - 1


def test_base_changes_seed() -> None:
    """A different base seed gives a different stream."""
    assert derive_seed(1, "x") != derive_seed(2, "x")


def test_label_order_matters() -> None:
    """Reordering labels gives a different seed."""
    assert derive_seed(7, "a", "b") != derive_seed(7, "b", "a")


def test_label_boundaries_are_unambiguous() -> None:
    """("a", "b") and ("ab",) must not collide because of string concatenation."""
    assert derive_seed(7, "a", "b") != derive_seed(7, "ab")


def test_integer_and_string_labels_agree() -> None:
    """Labels are converted with str(), so 8 and "8" name the same stream."""
    assert derive_seed(7, "perf", 8) == derive_seed(7, "perf", "8")


def test_seeds_spread_out() -> None:
    """Consecutive labels produce mostly distinct seeds (a weak sanity check on the hash)."""
    seeds = {derive_seed(1234, "request", index) for index in range(5000)}
    assert len(seeds) > 4990
