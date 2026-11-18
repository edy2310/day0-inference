"""Tests for the quality metrics: the top-k KL estimator, flips, divergence, and aggregation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from day0.quality.metrics import (
    ReferenceRecord,
    compute_item_metrics,
    first_divergence,
    flip_status,
    gen_kl_prefix_mean,
    top1_match,
    topk_kl,
)


def _exact_kl(p: np.ndarray, q: np.ndarray) -> float:
    """Exact KL divergence between two full distributions.

    Args:
        p: Reference probabilities.
        q: Candidate probabilities.

    Returns:
        ``sum p log(p / q)`` in nats.
    """
    return float(np.sum(p * np.log(p / q)))


def _as_topk(probs: np.ndarray, tokens: list[int] | None = None) -> dict[int, float]:
    """Turn probabilities into a top-k dictionary keyed by token id.

    Args:
        probs: Probabilities indexed by token id.
        tokens: Tokens to keep. Defaults to all of them.

    Returns:
        Token id to natural-log probability.
    """
    keep = tokens if tokens is not None else list(range(len(probs)))
    return {token: float(math.log(probs[token])) for token in keep}


def test_identical_distributions_have_zero_kl() -> None:
    """Identical top-k dictionaries give KL zero within 1e-12."""
    dist = {1: math.log(0.6), 2: math.log(0.3), 3: math.log(0.1)}
    assert topk_kl(dist, dist) == pytest.approx(0.0, abs=1e-12)


def test_full_vocabulary_matches_exact_kl() -> None:
    """With every token present, the estimator equals the exact KL to about 1e-8."""
    rng = np.random.default_rng(3)
    for _ in range(20):
        p = rng.dirichlet(np.ones(5))
        q = rng.dirichlet(np.ones(5))
        estimate = topk_kl(_as_topk(p), _as_topk(q))
        assert estimate == pytest.approx(_exact_kl(p, q), abs=1e-8)


def test_disjoint_top_k_sets_are_finite_and_positive() -> None:
    """Completely different top-k sets still give a finite, positive estimate."""
    p = {1: math.log(0.7), 2: math.log(0.2)}
    q = {8: math.log(0.5), 9: math.log(0.4)}
    value = topk_kl(p, q)
    assert math.isfinite(value)
    assert value > 0.0


def test_kl_is_never_negative_over_random_top_k() -> None:
    """Property test: 1000 seeded random top-k pairs never give a negative estimate."""
    rng = np.random.default_rng(99)
    for _ in range(1000):
        p = rng.dirichlet(np.full(30, 0.5))
        q = rng.dirichlet(np.full(30, 0.5))
        p_top = {int(t): float(math.log(p[t])) for t in np.argsort(p)[-10:]}
        q_top = {int(t): float(math.log(q[t])) for t in np.argsort(q)[-10:]}
        assert topk_kl(p_top, q_top) >= 0.0


def test_merge_step_is_a_lower_bound_of_exact_kl() -> None:
    """Merging outcomes outside a shared top-k set can only lower the KL (data processing).

    This test uses the same token set on both sides, so no imputation is involved. It checks the
    merge step alone.
    """
    rng = np.random.default_rng(11)
    for _ in range(300):
        p = rng.dirichlet(np.ones(20))
        q = rng.dirichlet(np.ones(20))
        subset = sorted(rng.choice(20, size=int(rng.integers(3, 15)), replace=False).tolist())
        estimate = topk_kl(_as_topk(p, subset), _as_topk(q, subset))
        assert estimate <= _exact_kl(p, q) + 1e-9


def test_top1_match_breaks_ties_by_lowest_id() -> None:
    """Tied maxima compare by the lowest token id on both sides."""
    tied_a = {5: math.log(0.4), 2: math.log(0.4), 9: math.log(0.2)}
    tied_b = {2: math.log(0.4), 7: math.log(0.4)}
    assert top1_match(tied_a, tied_b) is True
    assert (
        top1_match({5: math.log(0.6), 1: math.log(0.4)}, {1: math.log(0.6), 5: math.log(0.4)})
        is False
    )
    assert top1_match({}, {1: 0.0}) is False


def test_first_divergence_cases() -> None:
    """Divergence is the first differing position. A prefix diverges at the end of the shorter."""
    assert first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert first_divergence([1, 2, 3], [1, 9, 3]) == 1
    assert first_divergence([1, 2, 3], [1, 2]) == 2
    assert first_divergence([], []) is None
    assert first_divergence([], [4]) == 0


def test_flip_directions_and_missing_gold() -> None:
    """Flips are reported with direction, and items without gold have no flip."""
    assert flip_status(True, False) == (True, "correct_to_wrong")
    assert flip_status(False, True) == (True, "wrong_to_correct")
    assert flip_status(True, True) == (False, None)
    assert flip_status(None, True) == (None, None)


def test_gen_kl_prefix_stops_before_divergence() -> None:
    """Only positions before the first divergence are averaged, and position 0 is included."""
    same = {1: math.log(0.9), 2: math.log(0.1)}
    different = {1: math.log(0.5), 2: math.log(0.5)}
    ref_topk = [same, same, same]
    cand_topk = [same, same, different]
    value = gen_kl_prefix_mean(ref_topk, cand_topk, [1, 1, 1], [1, 1, 2])
    assert value == pytest.approx(0.0, abs=1e-12)
    assert gen_kl_prefix_mean(ref_topk, cand_topk, [1, 1, 1], [1, 1, 1]) > 0.0
    assert gen_kl_prefix_mean(ref_topk, cand_topk, [1, 1, 1], [9, 1, 1]) is None
    assert gen_kl_prefix_mean(None, cand_topk, [1], [1]) is None


def _reference(
    prefill: list[dict[int, float]] | None, lp: list[float] | None, correct: bool | None = True
) -> ReferenceRecord:
    """Build a hand-made reference record.

    Args:
        prefill: Prefill top-k per output position.
        lp: Reference logprob of each output token.
        correct: Reference correctness.

    Returns:
        The record.
    """
    return ReferenceRecord(
        item_id="item-1",
        suite="s",
        slice_index=0,
        prompt_ids=[256, 65],
        output_ids=[7, 8],
        output_text="ab",
        gold="72",
        answer_type="number",
        correct=correct,
        gen_topk=[{7: -0.1, 8: -2.0}, {8: -0.5, 9: -1.0}],
        prefill_topk=prefill,
        prefill_token_lp=lp,
    )


def test_item_metrics_for_identical_candidate() -> None:
    """An identical candidate has zero KL, full agreement, no divergence, and no flip."""
    prefill = [{7: math.log(0.9), 8: math.log(0.1)}, {8: math.log(0.8), 9: math.log(0.2)}]
    ref = _reference(prefill, [math.log(0.9), math.log(0.8)])
    metrics = compute_item_metrics(
        ref=ref,
        cand_output_ids=[7, 8],
        cand_output_text="ab",
        cand_gen_topk=ref.gen_topk,
        cand_prefill_topk=prefill,
        cand_prefill_token_lp=[math.log(0.9), math.log(0.8)],
        cand_correct=True,
    )
    assert metrics.prefill_kl_mean == pytest.approx(0.0, abs=1e-12)
    assert metrics.prefill_top1_agreement == 1.0
    assert metrics.prefill_nll_delta_mean == pytest.approx(0.0, abs=1e-12)
    assert metrics.exact_match is True
    assert metrics.first_divergence is None
    assert metrics.flip is False


def test_item_metrics_for_diverging_wrong_candidate() -> None:
    """A diverging, wrong candidate reports divergence, a correct-to-wrong flip, and NLL delta."""
    prefill = [{7: math.log(0.9), 8: math.log(0.1)}, {8: math.log(0.8), 9: math.log(0.2)}]
    ref = _reference(prefill, [math.log(0.9), math.log(0.8)])
    cand_prefill = [{7: math.log(0.5), 8: math.log(0.5)}, {8: math.log(0.8), 9: math.log(0.2)}]
    metrics = compute_item_metrics(
        ref=ref,
        cand_output_ids=[8, 8],
        cand_output_text="bb",
        cand_gen_topk=[{8: -0.2, 7: -1.8}, {8: -0.5, 9: -1.0}],
        cand_prefill_topk=cand_prefill,
        cand_prefill_token_lp=[math.log(0.5), math.log(0.8)],
        cand_correct=False,
    )
    assert metrics.exact_match is False
    assert metrics.first_divergence == 0
    assert metrics.first_divergence_frac == pytest.approx(0.0)
    assert metrics.gen_kl_prefix_mean is None
    assert metrics.flip is True
    assert metrics.flip_direction == "correct_to_wrong"
    assert metrics.prefill_kl_mean is not None and metrics.prefill_kl_mean > 0.0
    assert metrics.prefill_nll_delta_mean == pytest.approx((math.log(0.9) - math.log(0.5)) / 2.0)


def test_item_metrics_without_prefill_or_ids() -> None:
    """Without prefill data or ids, the corresponding metrics are None and text equality decides."""
    ref = _reference(None, None, correct=None)
    metrics = compute_item_metrics(
        ref=ref,
        cand_output_ids=None,
        cand_output_text="ab",
        cand_gen_topk=None,
        cand_prefill_topk=None,
        cand_prefill_token_lp=None,
        cand_correct=None,
    )
    assert metrics.prefill_kl_mean is None
    assert metrics.gen_kl_prefix_mean is None
    assert metrics.exact_match is True
    assert metrics.flip is None
