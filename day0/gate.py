"""Pass/fail evaluation, recommendation, and Pareto front (§14).

Every function here works on loaded artifacts and returns plain data. The same code runs during
``day0 run`` and during ``day0 report``, so a card always matches the artifacts it was built from.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from day0.artifacts import LoadedCandidate
from day0.config import BudgetConfig
from day0.quality.metrics import CandidateRecord
from day0.seeds import derive_seed
from day0.stats import bootstrap_mean_ci, wilson_interval

STATUS_ERROR = "ERROR"
STATUS_INSUFFICIENT = "INSUFFICIENT_DATA"
STATUS_FAIL_QUALITY = "FAIL_QUALITY"
STATUS_FAIL_SLO = "FAIL_SLO"
STATUS_PASS = "PASS"

GATED_METRICS = ("flip_rate", "prefill_kl_mean", "prefill_top1_agreement")


@dataclass(frozen=True)
class Interval:
    """A point estimate with its two-sided interval.

    Attributes:
        value: Point estimate, or ``None`` when there is no data.
        lower: Lower bound, or ``None``.
        upper: Upper bound, or ``None``.
        n: Number of items behind the estimate.
    """

    value: float | None
    lower: float | None
    upper: float | None
    n: int

    def as_dict(self) -> dict[str, float | int | None]:
        """Return the interval as a JSON-compatible dictionary.

        Returns:
            Fields ``value``, ``lower``, ``upper``, and ``n``. Non-finite bounds become ``None``.
        """
        return {
            "value": _finite(self.value),
            "lower": _finite(self.lower),
            "upper": _finite(self.upper),
            "n": self.n,
        }


def _finite(value: float | None) -> float | None:
    """Return the value, or ``None`` when it is missing or not finite.

    Args:
        value: A number or ``None``.

    Returns:
        The value when it is finite, otherwise ``None``.
    """
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def proportion_interval(successes: int, trials: int, confidence: float) -> Interval:
    """Wilson interval for a proportion.

    Args:
        successes: Number of successes.
        trials: Number of trials.
        confidence: Two-sided confidence level.

    Returns:
        The interval. Its value is ``None`` when there are no trials.
    """
    if trials == 0:
        return Interval(value=None, lower=None, upper=None, n=0)
    lower, upper = wilson_interval(successes, trials, confidence)
    return Interval(value=successes / trials, lower=lower, upper=upper, n=trials)


def mean_interval(
    values: Sequence[float], *, confidence: float, resamples: int, seed: int
) -> Interval:
    """Percentile bootstrap interval for a mean of per-prompt values.

    Args:
        values: One value per prompt.
        confidence: Two-sided confidence level.
        resamples: Number of resamples.
        seed: Bootstrap seed.

    Returns:
        The interval. Its value is ``None`` when there are no values.
    """
    if not values:
        return Interval(value=None, lower=None, upper=None, n=0)
    ci = bootstrap_mean_ci(values, confidence=confidence, resamples=resamples, seed=seed)
    return Interval(value=ci.mean, lower=ci.lower, upper=ci.upper, n=len(values))


def aggregate(
    records: Sequence[CandidateRecord],
    *,
    ref_correct: Mapping[str, bool | None],
    budget: BudgetConfig,
    run_seed: int,
    config_name: str,
    label: str,
) -> dict[str, Any]:
    """Aggregate per-item metrics over a set of items.

    Args:
        records: Candidate records for the items (overall, or one slice).
        ref_correct: Reference correctness by item id.
        budget: Confidence level and bootstrap settings.
        run_seed: Run seed, for the bootstrap streams.
        config_name: Candidate name, for the bootstrap streams.
        label: ``all`` or the slice index, for the bootstrap streams.

    Returns:
        Intervals for every metric, plus item counts.
    """

    def boot(metric: str, values: list[float]) -> Interval:
        seed = derive_seed(run_seed, "bootstrap", config_name, metric, label)
        return mean_interval(
            values, confidence=budget.confidence, resamples=budget.bootstrap_resamples, seed=seed
        )

    flips = [r.metrics for r in records if r.metrics.flip is not None]
    n_gold = len(flips)
    prefill_kls = [
        r.metrics.prefill_kl_mean for r in records if r.metrics.prefill_kl_mean is not None
    ]
    top1 = [
        r.metrics.prefill_top1_agreement
        for r in records
        if r.metrics.prefill_top1_agreement is not None
    ]
    nll = [
        r.metrics.prefill_nll_delta_mean
        for r in records
        if r.metrics.prefill_nll_delta_mean is not None
    ]
    gen_kl = [
        r.metrics.gen_kl_prefix_mean for r in records if r.metrics.gen_kl_prefix_mean is not None
    ]
    ref_gold = [ref_correct[r.item_id] for r in records if ref_correct.get(r.item_id) is not None]
    cand_gold = [r.correct for r in records if r.correct is not None]
    confidence = budget.confidence
    return {
        "n_items": len(records),
        "n_gold": n_gold,
        "n_prefill": len(prefill_kls),
        "n_gen_kl": len(gen_kl),
        "flip_rate": proportion_interval(
            sum(1 for m in flips if m.flip), n_gold, confidence
        ).as_dict(),
        "flip_rate_c2w": proportion_interval(
            sum(1 for m in flips if m.flip_direction == "correct_to_wrong"), n_gold, confidence
        ).as_dict(),
        "flip_rate_w2c": proportion_interval(
            sum(1 for m in flips if m.flip_direction == "wrong_to_correct"), n_gold, confidence
        ).as_dict(),
        "exact_match_rate": proportion_interval(
            sum(1 for r in records if r.metrics.exact_match), len(records), confidence
        ).as_dict(),
        "prefill_kl_mean": boot("prefill_kl_mean", prefill_kls).as_dict(),
        "prefill_top1_agreement": boot("prefill_top1_agreement", top1).as_dict(),
        "prefill_nll_delta_mean": boot("prefill_nll_delta_mean", nll).as_dict(),
        "gen_kl_prefix_mean": boot("gen_kl_prefix_mean", gen_kl).as_dict(),
        "accuracy_ref": proportion_interval(
            sum(1 for ok in ref_gold if ok), len(ref_gold), confidence
        ).as_dict(),
        "accuracy_cand": proportion_interval(
            sum(1 for ok in cand_gold if ok), len(cand_gold), confidence
        ).as_dict(),
    }


@dataclass
class CandidateEvaluation:
    """The gate result of one candidate.

    Attributes:
        name: Candidate name.
        status: One of the ``STATUS_*`` constants.
        reasons: Human-readable reasons for a non-pass status.
        error: Error message when the status is ``ERROR``.
        failures: Gate failures, one entry per (slice, metric).
        insufficient: Slice and metric pairs with too few items.
        not_covered: Slice and metric pairs with no data.
        overall: Aggregates over all items.
        slices: Aggregates and per-metric checks for each slice that has items.
        target: The target-concurrency perf summary, or ``None``.
    """

    name: str
    status: str
    reasons: list[str] = field(default_factory=list)
    error: str | None = None
    failures: list[dict[str, Any]] = field(default_factory=list)
    insufficient: list[dict[str, Any]] = field(default_factory=list)
    not_covered: list[dict[str, Any]] = field(default_factory=list)
    overall: dict[str, Any] = field(default_factory=dict)
    slices: list[dict[str, Any]] = field(default_factory=list)
    target: dict[str, Any] | None = None


def evaluate_candidate(
    *,
    name: str,
    candidate: LoadedCandidate | None,
    ref_correct: Mapping[str, bool | None],
    budget: BudgetConfig,
    slices: Sequence[int],
    run_seed: int,
    error_hint: str | None = None,
) -> CandidateEvaluation:
    """Evaluate one candidate against the budget and the SLO (§14.1).

    Statuses are checked in this order: ``ERROR``, ``INSUFFICIENT_DATA``, ``FAIL_QUALITY``,
    ``FAIL_SLO``, ``PASS``. The first one that applies is reported.

    Args:
        name: Candidate name.
        candidate: Loaded artifacts, or ``None`` when the candidate has none.
        ref_correct: Reference correctness by item id.
        budget: Thresholds and interval settings.
        slices: Context-length boundaries.
        run_seed: Run seed.
        error_hint: Message to use when artifacts are missing.

    Returns:
        The evaluation.
    """
    if candidate is None:
        return CandidateEvaluation(
            name=name,
            status=STATUS_ERROR,
            error=error_hint or "no artifacts for this candidate",
            reasons=[error_hint or "no artifacts for this candidate"],
        )
    if candidate.error is not None:
        return CandidateEvaluation(
            name=name,
            status=STATUS_ERROR,
            error=candidate.error,
            reasons=[candidate.error],
        )

    if not candidate.records:
        return CandidateEvaluation(
            name=name,
            status=STATUS_INSUFFICIENT,
            reasons=["no quality data was collected for this candidate"],
        )

    overall = aggregate(
        candidate.records,
        ref_correct=ref_correct,
        budget=budget,
        run_seed=run_seed,
        config_name=name,
        label="all",
    )
    evaluation = CandidateEvaluation(name=name, status=STATUS_PASS, overall=overall)

    for index in range(len(slices) - 1):
        bucket = [r for r in candidate.records if r.slice_index == index]
        bounds = [slices[index], slices[index + 1]]
        if not bucket:
            evaluation.not_covered.append({"slice": index, "bounds": bounds, "reason": "no items"})
            continue
        agg = aggregate(
            bucket,
            ref_correct=ref_correct,
            budget=budget,
            run_seed=run_seed,
            config_name=name,
            label=str(index),
        )
        checks: dict[str, str] = {}
        counts = {
            "flip_rate": agg["n_gold"],
            "prefill_kl_mean": agg["n_prefill"],
            "prefill_top1_agreement": agg["n_prefill"],
        }
        for metric in GATED_METRICS:
            count = counts[metric]
            if count == 0:
                checks[metric] = "not covered"
                evaluation.not_covered.append({"slice": index, "bounds": bounds, "metric": metric})
            elif count < budget.min_prompts_per_slice:
                checks[metric] = "insufficient data"
                evaluation.insufficient.append(
                    {
                        "slice": index,
                        "bounds": bounds,
                        "metric": metric,
                        "n": count,
                        "required": budget.min_prompts_per_slice,
                    }
                )
            else:
                checks[metric] = "pass"
        _check_slice_bounds(evaluation, agg, checks, budget, index, bounds)
        evaluation.slices.append(
            {
                "slice": index,
                "bounds": bounds,
                "n_items": agg["n_items"],
                "aggregate": agg,
                "checks": checks,
            }
        )

    perf_target = candidate.perf.get("target") if candidate.perf is not None else None
    if not isinstance(perf_target, dict):
        evaluation.target = None
        evaluation.reasons.append("no performance data at the target concurrency")
    else:
        evaluation.target = perf_target
        if perf_target.get("invalid") is True:
            evaluation.reasons.append(f"performance invalid: {perf_target.get('invalid_reason')}")

    if (
        evaluation.insufficient
        or evaluation.target is None
        or (evaluation.target is not None and evaluation.target.get("invalid") is True)
    ):
        evaluation.status = STATUS_INSUFFICIENT
    elif evaluation.failures:
        evaluation.status = STATUS_FAIL_QUALITY
        evaluation.reasons = [
            f"slice {f['slice']} {f['metric']}: bound {f['bound']:.4g} vs threshold "
            f"{f['threshold']:.4g}"
            for f in evaluation.failures
        ]
    elif evaluation.target is None or evaluation.target.get("slo_met") is not True:
        evaluation.status = STATUS_FAIL_SLO
        evaluation.reasons = ["p99 latency at the target concurrency does not meet the SLO"]
    else:
        evaluation.status = STATUS_PASS
        evaluation.reasons = []
    return evaluation


def _check_slice_bounds(
    evaluation: CandidateEvaluation,
    agg: Mapping[str, Any],
    checks: dict[str, str],
    budget: BudgetConfig,
    index: int,
    bounds: list[int],
) -> None:
    """Compare a slice's bounds with the thresholds and record failures.

    Only metrics whose check is ``pass`` are compared. The one-sided bound used is the bound of
    the two-sided interval on the relevant side (§14.1).

    Args:
        evaluation: Evaluation to append failures to.
        agg: Aggregates of the slice.
        checks: Per-metric check results, updated in place.
        budget: Thresholds.
        index: Slice index.
        bounds: Slice boundaries.
    """
    tests = (
        ("flip_rate", "upper", budget.flip_rate_max),
        ("prefill_kl_mean", "upper", budget.kl_mean_max),
        ("prefill_top1_agreement", "lower", budget.top1_agreement_min),
    )
    for metric, side, threshold in tests:
        if checks.get(metric) != "pass":
            continue
        interval = agg[metric]
        bound = interval.get(side)
        if bound is None:
            continue
        failed = bound > threshold if side == "upper" else bound < threshold
        if failed:
            checks[metric] = "fail"
            evaluation.failures.append(
                {
                    "slice": index,
                    "bounds": bounds,
                    "metric": metric,
                    "side": side,
                    "bound": bound,
                    "threshold": threshold,
                }
            )


@dataclass(frozen=True)
class Recommendation:
    """The recommended configuration, or the reason there is none.

    Attributes:
        name: Recommended configuration, or ``None``.
        reason: Why there is no recommendation, or ``None``.
    """

    name: str | None
    reason: str | None


NO_RECOMMENDATION_REASON = "no configuration meets the SLO at target concurrency"


def recommend(
    *,
    reference_name: str,
    reference_target: Mapping[str, Any] | None,
    evaluations: Sequence[CandidateEvaluation],
) -> Recommendation:
    """Pick the best configuration that passes the gates and meets the SLO (§14.2).

    Eligible configurations are the reference (if its SLO is met) and every candidate with status
    ``PASS``. The highest target throughput wins. Ties go to the lower p99 TTFT, then the lower
    cost.

    Args:
        reference_name: Name of the reference configuration.
        reference_target: The reference target-concurrency summary, or ``None``.
        evaluations: Candidate evaluations.

    Returns:
        The recommendation.
    """
    eligible: list[tuple[str, Mapping[str, Any]]] = []
    if reference_target is not None and reference_target.get("slo_met") is True:
        eligible.append((reference_name, reference_target))
    for evaluation in evaluations:
        if (
            evaluation.status == STATUS_PASS
            and evaluation.target is not None
            and evaluation.target.get("slo_met") is True
        ):
            eligible.append((evaluation.name, evaluation.target))
    if not eligible:
        return Recommendation(name=None, reason=NO_RECOMMENDATION_REASON)

    def key(item: tuple[str, Mapping[str, Any]]) -> tuple[float, float, float]:
        _, target = item
        throughput = float(target.get("output_tok_per_s") or 0.0)
        ttft = target.get("ttft_ms")
        p99_ttft = (
            float(p99)
            if isinstance(ttft, dict) and (p99 := ttft.get("p99")) is not None
            else math.inf
        )
        cost = target.get("cost_per_1m_output_tokens_usd")
        cost_value = float(cost) if cost is not None else math.inf
        return (-throughput, p99_ttft, cost_value)

    best = min(eligible, key=key)
    return Recommendation(name=best[0], reason=None)


@dataclass(frozen=True)
class ParetoPoint:
    """A configuration in the quality-versus-throughput plane.

    Attributes:
        name: Configuration name.
        throughput_ratio: Target throughput divided by the reference's.
        flip_rate: Point estimate of the flip rate.
        flip_upper: Upper Wilson bound of the flip rate, used for dominance.
    """

    name: str
    throughput_ratio: float
    flip_rate: float
    flip_upper: float


def pareto_front(points: Sequence[ParetoPoint]) -> list[str]:
    """Names of the points that no other point dominates (§14.4).

    Point ``q`` dominates ``p`` when ``q.x >= p.x`` and ``q.y <= p.y``, with at least one strict
    inequality. Here ``x`` is the throughput ratio and ``y`` is the upper flip bound.

    Args:
        points: Configurations with valid data.

    Returns:
        Names on the front, in input order.
    """
    front: list[str] = []
    for point in points:
        dominated = any(
            other is not point
            and other.throughput_ratio >= point.throughput_ratio
            and other.flip_upper <= point.flip_upper
            and (
                other.throughput_ratio > point.throughput_ratio
                or other.flip_upper < point.flip_upper
            )
            for other in points
        )
        if not dominated:
            front.append(point.name)
    return front
