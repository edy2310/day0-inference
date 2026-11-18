"""Tests for the gate: every status path, slice coverage, recommendation, and the Pareto front."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from day0.artifacts import LoadedCandidate
from day0.config import BudgetConfig
from day0.gate import (
    STATUS_ERROR,
    STATUS_FAIL_QUALITY,
    STATUS_FAIL_SLO,
    STATUS_INSUFFICIENT,
    STATUS_PASS,
    ParetoPoint,
    evaluate_candidate,
    pareto_front,
    recommend,
)
from day0.quality.metrics import CandidateRecord, ItemMetrics

SLICES = [0, 512, 2048, 65536]
BUDGET = BudgetConfig(
    flip_rate_max=0.05,
    kl_mean_max=0.01,
    top1_agreement_min=0.95,
    min_prompts_per_slice=30,
    confidence=0.95,
    bootstrap_resamples=200,
)


def _metrics(
    *, flip: bool | None, direction: str | None = None, kl: float = 0.0, top1: float = 1.0
) -> ItemMetrics:
    """Build per-item metrics with the fields the gate reads.

    Args:
        flip: Whether correctness changed, or ``None`` without gold.
        direction: Flip direction.
        kl: Prefill KL mean for the item.
        top1: Prefill top-1 agreement for the item.

    Returns:
        The metrics.
    """
    return ItemMetrics(
        prefill_kl_mean=kl,
        prefill_kl_p99_position=kl,
        prefill_top1_agreement=top1,
        prefill_nll_delta_mean=0.0,
        exact_match=not flip if flip is not None else True,
        first_divergence=None,
        first_divergence_frac=None,
        gen_kl_prefix_mean=kl,
        flip=flip,
        flip_direction=direction,
    )


def _records(
    count: int,
    *,
    slice_index: int = 0,
    wrong: int = 0,
    kl: float = 0.0,
    top1: float = 1.0,
    prefix: str = "item",
) -> list[CandidateRecord]:
    """Build ``count`` candidate records, the first ``wrong`` of which flip to wrong.

    Args:
        count: Number of items.
        slice_index: Slice of every item.
        wrong: Number of correct-to-wrong flips.
        kl: Prefill KL per item.
        top1: Top-1 agreement per item.
        prefix: Item id prefix.

    Returns:
        The records.
    """
    records: list[CandidateRecord] = []
    for index in range(count):
        flipped = index < wrong
        records.append(
            CandidateRecord(
                item_id=f"{prefix}-{slice_index}-{index}",
                suite="s",
                slice_index=slice_index,
                output_ids=[1],
                output_text="x",
                correct=not flipped,
                gen_topk=None,
                prefill_topk=None,
                prefill_token_lp=None,
                metrics=_metrics(
                    flip=flipped,
                    direction="correct_to_wrong" if flipped else None,
                    kl=kl,
                    top1=top1,
                ),
            )
        )
    return records


def _ref_correct(records: list[CandidateRecord]) -> dict[str, bool | None]:
    """Reference correctness that matches the candidate records (all correct).

    Args:
        records: Candidate records.

    Returns:
        Reference correctness by item id.
    """
    return {record.item_id: True for record in records}


def _perf(slo_met: bool = True, invalid: bool = False) -> dict[str, Any]:
    """Build a perf document with only the target summary the gate reads.

    Args:
        slo_met: Whether the target met the SLO.
        invalid: Whether the target is invalid.

    Returns:
        The perf document.
    """
    return {
        "target": {
            "slo_met": slo_met,
            "invalid": invalid,
            "invalid_reason": "bad" if invalid else None,
            "output_tok_per_s": 100.0,
            "ttft_ms": {"p99": 10.0},
            "tpot_ms": {"p99": 1.0},
            "cost_per_1m_output_tokens_usd": 1.0,
        }
    }


def _candidate(
    records: list[CandidateRecord],
    *,
    perf: dict[str, Any] | None = None,
    status: dict[str, Any] | None = None,
) -> LoadedCandidate:
    """Build loaded candidate artifacts.

    Args:
        records: Quality records.
        perf: Perf document, or ``None`` to omit it.
        status: Status document.

    Returns:
        The loaded candidate.
    """
    return LoadedCandidate(
        name="cand",
        path=Path("unused"),
        records=records,
        perf=perf if perf is not None else _perf(),
        determinism=None,
        status=status or {"status": "COMPLETED"},
    )


def _evaluate(candidate: LoadedCandidate | None, records: list[CandidateRecord]) -> Any:
    """Evaluate a candidate with the test budget and slices.

    Args:
        candidate: Loaded candidate, or ``None``.
        records: Records, used for the reference correctness map.

    Returns:
        The evaluation.
    """
    return evaluate_candidate(
        name="cand",
        candidate=candidate,
        ref_correct=_ref_correct(records),
        budget=BUDGET,
        slices=SLICES,
        run_seed=1,
    )


def test_pass_when_quality_and_slo_hold() -> None:
    """Identical outputs and a met SLO give PASS."""
    records = _records(100, slice_index=0)
    evaluation = _evaluate(_candidate(records), records)
    assert evaluation.status == STATUS_PASS
    assert evaluation.failures == []


def test_zero_flips_in_small_sample_fails_the_wilson_bound() -> None:
    """Zero flips in 60 items still fail at 5%, because the Wilson upper bound is about 6%.

    This shows the gate is conservative for small samples, which is intended.
    """
    records = _records(60, slice_index=0)
    evaluation = _evaluate(_candidate(records), records)
    assert evaluation.status == STATUS_FAIL_QUALITY
    assert evaluation.failures[0]["metric"] == "flip_rate"


def test_zero_flips_in_100_items_passes_the_wilson_bound() -> None:
    """Zero flips in 100 items passes, because the Wilson upper bound is about 3.7%."""
    records = _records(100, slice_index=0)
    evaluation = _evaluate(_candidate(records), records)
    assert evaluation.status == STATUS_PASS


def test_many_flips_fail_quality_and_name_the_bound() -> None:
    """Twenty flips in 100 items fail, and the failure names the metric, slice, and bound.

    Args: none; the values are chosen by hand.
    """
    records = _records(100, slice_index=0, wrong=20)
    evaluation = _evaluate(_candidate(records), records)
    assert evaluation.status == STATUS_FAIL_QUALITY
    failure = evaluation.failures[0]
    assert failure["metric"] == "flip_rate"
    assert failure["side"] == "upper"
    assert failure["bound"] > BUDGET.flip_rate_max


def test_large_kl_fails_quality() -> None:
    """A mean prefill KL above the threshold fails, with the upper bound reported."""
    records = _records(100, slice_index=0, kl=0.5)
    evaluation = _evaluate(_candidate(records), records)
    assert evaluation.status == STATUS_FAIL_QUALITY
    assert any(f["metric"] == "prefill_kl_mean" for f in evaluation.failures)


def test_small_slice_is_insufficient_data() -> None:
    """A slice with fewer items than ``min_prompts_per_slice`` is INSUFFICIENT_DATA."""
    records = _records(10, slice_index=0)
    evaluation = _evaluate(_candidate(records), records)
    assert evaluation.status == STATUS_INSUFFICIENT
    assert {item["metric"] for item in evaluation.insufficient} >= {"flip_rate"}


def test_slo_failure_when_quality_passes() -> None:
    """Quality passes, but the target misses the SLO, so the status is FAIL_SLO."""
    records = _records(100, slice_index=0)
    evaluation = _evaluate(_candidate(records, perf=_perf(slo_met=False)), records)
    assert evaluation.status == STATUS_FAIL_SLO


def test_errors_and_missing_artifacts_are_reported() -> None:
    """A candidate with an error, or without artifacts, has status ERROR and a message."""
    records = _records(100, slice_index=0)
    failed = _candidate(records, status={"status": "ERROR", "error": "boom"})
    evaluation = _evaluate(failed, records)
    assert evaluation.status == STATUS_ERROR
    assert evaluation.error == "boom"
    missing = evaluate_candidate(
        name="cand",
        candidate=None,
        ref_correct={},
        budget=BUDGET,
        slices=SLICES,
        run_seed=1,
        error_hint="crashed early",
    )
    assert missing.status == STATUS_ERROR
    assert missing.error == "crashed early"


def test_slices_without_items_are_listed_as_not_covered() -> None:
    """Empty slices are not evaluated, and they are reported as not covered.

    Args: none; the values are chosen by hand.
    """
    records = _records(100, slice_index=1)
    evaluation = _evaluate(_candidate(records), records)
    assert evaluation.status == STATUS_PASS
    uncovered = {item["slice"] for item in evaluation.not_covered if "metric" not in item}
    assert uncovered == {0, 2}
    assert [entry["slice"] for entry in evaluation.slices] == [1]


def test_recommendation_prefers_throughput_then_latency_then_cost() -> None:
    """The highest throughput wins; ties go to the lower p99 TTFT, then to the lower cost."""

    def evaluation(name: str, tok: float, ttft: float, cost: float) -> Any:
        target = {
            "slo_met": True,
            "output_tok_per_s": tok,
            "ttft_ms": {"p99": ttft},
            "cost_per_1m_output_tokens_usd": cost,
        }
        return type("Eval", (), {"name": name, "status": STATUS_PASS, "target": target})()

    result = recommend(
        reference_name="ref",
        reference_target={
            "slo_met": True,
            "output_tok_per_s": 90.0,
            "ttft_ms": {"p99": 9.0},
            "cost_per_1m_output_tokens_usd": 3.0,
        },
        evaluations=[
            evaluation("a", 120.0, 8.0, 2.0),
            evaluation("b", 120.0, 7.0, 4.0),
            evaluation("c", 120.0, 7.0, 1.0),
        ],
    )
    assert result.name == "c"
    assert result.reason is None


def test_no_recommendation_when_nothing_meets_the_slo() -> None:
    """With no SLO-compliant configuration, the reason is the one the specification gives."""
    result = recommend(
        reference_name="ref",
        reference_target={"slo_met": False},
        evaluations=[],
    )
    assert result.name is None
    assert result.reason == "no configuration meets the SLO at target concurrency"


def test_pareto_front_drops_dominated_points() -> None:
    """A point is dominated when another has at least its throughput and at most its flip bound."""
    points = [
        ParetoPoint("ref", 1.0, 0.0, 0.0),
        ParetoPoint("fast-ok", 1.3, 0.0, 0.02),
        ParetoPoint("dominated", 1.0, 0.0, 0.03),
        ParetoPoint("tie-speed", 1.3, 0.0, 0.02),
        ParetoPoint("fast-noisy", 1.6, 0.2, 0.27),
    ]
    front = pareto_front(points)
    assert front == ["ref", "fast-ok", "tie-speed", "fast-noisy"]
