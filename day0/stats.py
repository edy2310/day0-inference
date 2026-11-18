"""Pure statistics: Wilson intervals, percentile bootstrap, and percentiles.

Every function takes plain numbers and returns plain numbers. Random draws use an explicit seed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np


@dataclass(frozen=True)
class BootstrapCI:
    """Mean of a sample and its percentile bootstrap interval.

    Attributes:
        mean: Sample mean (not the bootstrap mean).
        lower: Lower bound of the two-sided interval.
        upper: Upper bound of the two-sided interval.
    """

    mean: float
    lower: float
    upper: float


def z_for_confidence(confidence: float) -> float:
    """Return the two-sided normal quantile for a confidence level.

    Args:
        confidence: Two-sided confidence level, between 0 and 1.

    Returns:
        ``z`` such that ``P(-z < Z < z) = confidence``.
    """
    return NormalDist().inv_cdf((1.0 + confidence) / 2.0)


def wilson_interval(successes: int, trials: int, confidence: float) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Args:
        successes: Number of successes, between 0 and ``trials``.
        trials: Number of trials.
        confidence: Two-sided confidence level.

    Returns:
        ``(lower, upper)``. Returns ``(nan, nan)`` when there are no trials.

    Raises:
        ValueError: If ``successes`` is outside ``[0, trials]``.
    """
    if trials <= 0:
        return (math.nan, math.nan)
    if not 0 <= successes <= trials:
        raise ValueError(f"successes={successes} must be between 0 and trials={trials}")
    z = z_for_confidence(confidence)
    z2 = z * z
    p = successes / trials
    denominator = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denominator
    half_width = z * math.sqrt(p * (1.0 - p) / trials + z2 / (4.0 * trials * trials)) / denominator
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def bootstrap_mean_ci(
    values: Sequence[float] | np.ndarray,
    *,
    confidence: float,
    resamples: int,
    seed: int,
) -> BootstrapCI:
    """Percentile bootstrap interval for the mean of independent values.

    The unit of resampling is the value itself. Callers pass one value per prompt, so positions
    within a prompt never act as independent samples.

    Args:
        values: Sample values, one per independent unit.
        confidence: Two-sided confidence level.
        resamples: Number of bootstrap resamples.
        seed: Seed for the resampling generator.

    Returns:
        The sample mean and the interval. All fields are ``nan`` for an empty sample.
    """
    array = np.asarray(values, dtype=np.float64)
    n = array.shape[0]
    if n == 0:
        return BootstrapCI(mean=math.nan, lower=math.nan, upper=math.nan)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(resamples, n))
    resampled_means = array[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.percentile(
        resampled_means, [100.0 * alpha, 100.0 * (1.0 - alpha)], method="linear"
    )
    return BootstrapCI(mean=float(array.mean()), lower=float(lower), upper=float(upper))


def percentile(values: Sequence[float] | np.ndarray, q: float) -> float | None:
    """Linear-interpolation percentile, as ``numpy.percentile(method="linear")``.

    Args:
        values: Sample values.
        q: Percentile between 0 and 100.

    Returns:
        The percentile, or ``None`` for an empty sample.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.shape[0] == 0:
        return None
    return float(np.percentile(array, q, method="linear"))


def mean_or_none(values: Sequence[float]) -> float | None:
    """Arithmetic mean of a list, or ``None`` when the list is empty.

    Args:
        values: Values to average.

    Returns:
        The mean, or ``None``.
    """
    if not values:
        return None
    return float(sum(values) / len(values))
