"""Tests for the statistics helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from day0.stats import bootstrap_mean_ci, percentile, wilson_interval, z_for_confidence


def test_z_value_for_95_percent() -> None:
    """The two-sided 95% quantile is about 1.96."""
    assert z_for_confidence(0.95) == pytest.approx(1.959964, abs=1e-5)


@pytest.mark.parametrize(
    ("successes", "expected"),
    [
        (0, (0.0, 0.113513)),
        (15, (0.331541, 0.668459)),
        (30, (0.886487, 1.0)),
    ],
)
def test_wilson_matches_known_values(successes: int, expected: tuple[float, float]) -> None:
    """Wilson intervals for n = 30 at 95% match the reference values.

    Args:
        successes: Number of successes out of 30.
        expected: Expected ``(lower, upper)``.
    """
    lower, upper = wilson_interval(successes, 30, 0.95)
    assert lower == pytest.approx(expected[0], abs=1e-5)
    assert upper == pytest.approx(expected[1], abs=1e-5)


def test_wilson_empty_sample_is_nan() -> None:
    """With no trials the interval is undefined, so both bounds are NaN."""
    lower, upper = wilson_interval(0, 0, 0.95)
    assert math.isnan(lower) and math.isnan(upper)


def test_wilson_rejects_impossible_counts() -> None:
    """More successes than trials is a programming error and raises."""
    with pytest.raises(ValueError):
        wilson_interval(5, 3, 0.95)


def test_bootstrap_is_reproducible_with_a_seed() -> None:
    """The same seed gives the same interval, and a different seed usually does not."""
    values = np.linspace(0.0, 1.0, 50)
    first = bootstrap_mean_ci(values, confidence=0.95, resamples=500, seed=7)
    second = bootstrap_mean_ci(values, confidence=0.95, resamples=500, seed=7)
    other = bootstrap_mean_ci(values, confidence=0.95, resamples=500, seed=8)
    assert first == second
    assert (first.lower, first.upper) != (other.lower, other.upper)
    assert first.mean == pytest.approx(0.5)


def test_bootstrap_empty_sample_is_nan() -> None:
    """An empty sample gives NaN for every field."""
    result = bootstrap_mean_ci([], confidence=0.95, resamples=10, seed=1)
    assert math.isnan(result.mean) and math.isnan(result.lower) and math.isnan(result.upper)


def test_bootstrap_interval_coverage_is_close_to_nominal() -> None:
    """A 95% bootstrap interval covers the true mean in roughly 93% to 97% of seeded trials."""
    rng = np.random.default_rng(2024)
    covered = 0
    trials = 400
    for trial in range(trials):
        sample = rng.normal(loc=0.0, scale=1.0, size=40)
        result = bootstrap_mean_ci(sample, confidence=0.95, resamples=1000, seed=trial)
        covered += int(result.lower <= 0.0 <= result.upper)
    assert 0.93 <= covered / trials <= 0.97


def test_percentile_linear_and_empty() -> None:
    """Percentiles use linear interpolation, and an empty sample returns None."""
    assert percentile([1.0, 2.0, 3.0, 4.0], 50.0) == pytest.approx(2.5)
    assert percentile([], 99.0) is None
