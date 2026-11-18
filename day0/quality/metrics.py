"""Quality metrics: top-k KL estimator, top-1 agreement, flips, and divergence.

Every function is pure: it takes records or distributions and returns numbers. The estimator
follows the algorithm in §12.3 of the specification exactly. Rationale is in ``docs/metrics.md``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from day0.parsing import TopK
from day0.stats import percentile

KL_EPSILON = 1e-10


@dataclass(frozen=True)
class ReferenceRecord:
    """Everything the reference run stores for one item.

    Attributes:
        item_id: Item id.
        suite: Suite name.
        slice_index: Context-length slice of the prompt.
        prompt_ids: Prompt token ids.
        output_ids: Greedy output token ids.
        output_text: Greedy output text.
        gold: Gold answer, or ``None``.
        answer_type: ``number``, ``exact``, or ``None``.
        correct: Whether the output matches the gold answer, or ``None``.
        gen_topk: Top-k distribution at each output position (generation path).
        prefill_topk: Top-k distribution at each output position (teacher-forced path), or ``None``
            when the prefill path was skipped.
        prefill_token_lp: Reference logprob of each output token (teacher-forced path).
    """

    item_id: str
    suite: str
    slice_index: int
    prompt_ids: list[int]
    output_ids: list[int]
    output_text: str
    gold: str | None
    answer_type: str | None
    correct: bool | None
    gen_topk: list[TopK] | None
    prefill_topk: list[TopK] | None
    prefill_token_lp: list[float] | None


@dataclass(frozen=True)
class ItemMetrics:
    """Per-prompt metrics for one candidate.

    Attributes:
        prefill_kl_mean: Mean top-k KL over output positions, teacher-forced path.
        prefill_kl_p99_position: 99th percentile of the per-position KL.
        prefill_top1_agreement: Fraction of positions whose argmax matches.
        prefill_nll_delta_mean: Mean of ``reference_lp - candidate_lp`` over output tokens.
            Positive values mean the candidate finds the reference continuation less likely.
        exact_match: Whether the output token ids (or text, if ids are unavailable) match.
        first_divergence: First output position where the outputs differ, or ``None``.
        first_divergence_frac: ``first_divergence`` divided by the reference length.
        gen_kl_prefix_mean: Mean top-k KL on the common prefix, generation path.
        flip: Whether correctness changed, or ``None`` when either side has no gold label.
        flip_direction: ``correct_to_wrong``, ``wrong_to_correct``, or ``None``.
    """

    prefill_kl_mean: float | None
    prefill_kl_p99_position: float | None
    prefill_top1_agreement: float | None
    prefill_nll_delta_mean: float | None
    exact_match: bool
    first_divergence: int | None
    first_divergence_frac: float | None
    gen_kl_prefix_mean: float | None
    flip: bool | None
    flip_direction: str | None


@dataclass(frozen=True)
class CandidateRecord:
    """What a candidate run stores for one item: raw candidate data and its metrics.

    Attributes:
        item_id: Item id.
        suite: Suite name.
        slice_index: Context-length slice of the prompt.
        output_ids: Candidate output token ids, or ``None`` when the server did not return them.
        output_text: Candidate output text.
        correct: Whether the candidate output matches the gold answer, or ``None``.
        gen_topk: Candidate top-k per output position, or ``None`` when logprobs were unavailable.
        prefill_topk: Candidate top-k per reference output position, or ``None``.
        prefill_token_lp: Candidate logprob of each reference output token, or ``None``.
        metrics: Per-item metrics computed against the reference.
    """

    item_id: str
    suite: str
    slice_index: int
    output_ids: list[int] | None
    output_text: str
    correct: bool | None
    gen_topk: list[TopK] | None
    prefill_topk: list[TopK] | None
    prefill_token_lp: list[float] | None
    metrics: ItemMetrics


def _side_distribution(
    known: Mapping[int, float], universe: Sequence[int], eps: float
) -> list[float]:
    """Turn one side's top-k into a distribution over ``universe`` plus a TAIL bucket.

    Tokens in ``universe`` that this side does not list are imputed (see §12.3). The result has
    ``len(universe) + 1`` entries, and the last one is TAIL. Values are clipped to ``eps`` and
    renormalized.

    Args:
        known: Token id to log-probability for the tokens this side lists.
        universe: Union of the token ids of both sides, in a fixed order.
        eps: Clipping floor.

    Returns:
        Probabilities that sum to one.
    """
    probs = {token: math.exp(logprob) for token, logprob in known.items()}
    tail = max(0.0, 1.0 - sum(probs.values()))
    floor = min(probs.values()) if probs else 0.0
    missing = [token for token in universe if token not in probs]
    imputed = min(floor, tail / len(missing)) if missing else 0.0
    values = [probs.get(token, imputed) for token in universe]
    values.append(max(eps, tail - imputed * len(missing)))
    values = [max(value, eps) for value in values]
    total = sum(values)
    return [value / total for value in values]


def topk_kl(p_lp: Mapping[int, float], q_lp: Mapping[int, float], eps: float = KL_EPSILON) -> float:
    """Estimate ``KL(P || Q)`` in nats from two top-k distributions.

    The estimator merges outcomes that appear in neither top-k set into a TAIL bucket, imputes
    the mass of tokens missing from one side (``min(floor, tail / missing)``), clips to ``eps``,
    and renormalizes. The result is never negative.

    Args:
        p_lp: Reference top-k, token id to natural-log probability.
        q_lp: Candidate top-k, token id to natural-log probability.
        eps: Clipping floor for probabilities.

    Returns:
        The KL estimate, zero or positive.
    """
    universe = sorted(set(p_lp) | set(q_lp))
    if not universe:
        return 0.0
    p = _side_distribution(p_lp, universe, eps)
    q = _side_distribution(q_lp, universe, eps)
    kl = 0.0
    for p_value, q_value in zip(p, q, strict=True):
        kl += p_value * math.log(p_value / q_value)
    return max(0.0, kl)


def _argmax(distribution: Mapping[int, float]) -> int:
    """Return the most likely token. Ties go to the lowest id.

    Args:
        distribution: Token id to log-probability.

    Returns:
        The argmax token id.
    """
    return min(distribution.items(), key=lambda item: (-item[1], item[0]))[0]


def top1_match(p_lp: Mapping[int, float], q_lp: Mapping[int, float]) -> bool:
    """Whether two top-k distributions have the same argmax.

    Ties are broken by the lowest token id on both sides. Empty distributions never match.

    Args:
        p_lp: Reference top-k.
        q_lp: Candidate top-k.

    Returns:
        True when the argmax tokens are equal.
    """
    if not p_lp or not q_lp:
        return False
    return _argmax(p_lp) == _argmax(q_lp)


def first_divergence(ref_ids: Sequence[int], cand_ids: Sequence[int]) -> int | None:
    """First position at which two outputs differ.

    If one output is a prefix of the other, the divergence is at the end of the shorter one.

    Args:
        ref_ids: Reference output token ids.
        cand_ids: Candidate output token ids.

    Returns:
        The position, or ``None`` when the outputs are identical.
    """
    for position, (ref_token, cand_token) in enumerate(zip(ref_ids, cand_ids, strict=False)):
        if ref_token != cand_token:
            return position
    if len(ref_ids) != len(cand_ids):
        return min(len(ref_ids), len(cand_ids))
    return None


def flip_status(
    ref_correct: bool | None, cand_correct: bool | None
) -> tuple[bool | None, str | None]:
    """Whether correctness changed between reference and candidate, and in which direction.

    Args:
        ref_correct: Reference correctness, or ``None`` without a gold label.
        cand_correct: Candidate correctness, or ``None`` without a gold label.

    Returns:
        ``(flip, direction)``. Both are ``None`` when a side has no gold label.
    """
    if ref_correct is None or cand_correct is None:
        return (None, None)
    if ref_correct == cand_correct:
        return (False, None)
    return (True, "correct_to_wrong" if ref_correct else "wrong_to_correct")


def gen_kl_prefix_mean(
    ref_topk: Sequence[Mapping[int, float]] | None,
    cand_topk: Sequence[Mapping[int, float]] | None,
    ref_ids: Sequence[int],
    cand_ids: Sequence[int],
) -> float | None:
    """Mean top-k KL over the positions before the first divergence.

    Position 0 is always included when it exists, because its context is identical.

    Args:
        ref_topk: Reference generation top-k per output position.
        cand_topk: Candidate generation top-k per output position.
        ref_ids: Reference output token ids.
        cand_ids: Candidate output token ids.

    Returns:
        The mean KL, or ``None`` when no comparable position exists or logprobs are missing.
    """
    if ref_topk is None or cand_topk is None:
        return None
    divergence = first_divergence(ref_ids, cand_ids)
    limit = divergence if divergence is not None else min(len(ref_topk), len(cand_topk))
    positions = min(limit, len(ref_topk), len(cand_topk))
    if positions <= 0:
        return None
    values = [topk_kl(ref_topk[j], cand_topk[j]) for j in range(positions)]
    return sum(values) / len(values)


def compute_item_metrics(
    *,
    ref: ReferenceRecord,
    cand_output_ids: Sequence[int] | None,
    cand_output_text: str,
    cand_gen_topk: Sequence[Mapping[int, float]] | None,
    cand_prefill_topk: Sequence[Mapping[int, float]] | None,
    cand_prefill_token_lp: Sequence[float] | None,
    cand_correct: bool | None,
) -> ItemMetrics:
    """Compute every per-item metric for one candidate against its reference record.

    Args:
        ref: The reference record for the item.
        cand_output_ids: Candidate output ids, or ``None`` when unavailable.
        cand_output_text: Candidate output text.
        cand_gen_topk: Candidate generation top-k, or ``None``.
        cand_prefill_topk: Candidate top-k at the reference output positions, or ``None``.
        cand_prefill_token_lp: Candidate logprob of each reference output token, or ``None``.
        cand_correct: Candidate correctness, or ``None``.

    Returns:
        The per-item metrics.
    """
    prefill_kl_mean: float | None = None
    prefill_kl_p99: float | None = None
    prefill_top1: float | None = None
    prefill_nll: float | None = None
    if ref.prefill_topk is not None and cand_prefill_topk is not None:
        kls = [topk_kl(r, c) for r, c in zip(ref.prefill_topk, cand_prefill_topk, strict=True)]
        matches = [
            top1_match(r, c) for r, c in zip(ref.prefill_topk, cand_prefill_topk, strict=True)
        ]
        if kls:
            prefill_kl_mean = sum(kls) / len(kls)
            prefill_kl_p99 = percentile(kls, 99.0)
            prefill_top1 = sum(matches) / len(matches)
    if (
        ref.prefill_token_lp is not None
        and cand_prefill_token_lp is not None
        and ref.prefill_token_lp
    ):
        deltas = [r - c for r, c in zip(ref.prefill_token_lp, cand_prefill_token_lp, strict=True)]
        prefill_nll = sum(deltas) / len(deltas)

    if cand_output_ids is not None:
        exact = list(cand_output_ids) == ref.output_ids
        divergence: int | None = first_divergence(ref.output_ids, cand_output_ids)
    else:
        exact = cand_output_text == ref.output_text
        divergence = None
    divergence_frac = divergence / max(len(ref.output_ids), 1) if divergence is not None else None
    gen_kl = (
        gen_kl_prefix_mean(
            ref.gen_topk,
            cand_gen_topk,
            ref.output_ids,
            cand_output_ids if cand_output_ids is not None else [],
        )
        if cand_output_ids is not None
        else None
    )
    flip, direction = flip_status(ref.correct, cand_correct)
    return ItemMetrics(
        prefill_kl_mean=prefill_kl_mean,
        prefill_kl_p99_position=prefill_kl_p99,
        prefill_top1_agreement=prefill_top1,
        prefill_nll_delta_mean=prefill_nll,
        exact_match=exact,
        first_divergence=divergence,
        first_divergence_frac=divergence_frac,
        gen_kl_prefix_mean=gen_kl,
        flip=flip,
        flip_direction=direction,
    )
