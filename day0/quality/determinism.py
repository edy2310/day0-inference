"""Self-consistency checks: the noise floor (reference) and determinism (candidate).

Both checks run the generation path twice on the same prompts with two batch concurrencies, then
compare the outputs. They show how much a configuration differs from itself only because batch
composition changed, which is the baseline a candidate's differences should be read against.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from day0.client import CompletionResult, InferenceClient
from day0.quality.metrics import flip_status, gen_kl_prefix_mean
from day0.quality.runner import GenerationMode, GenRequest, generate_outputs
from day0.stats import wilson_interval
from day0.suites import check_answer


@dataclass(frozen=True)
class ConsistencyReport:
    """Comparison of two generation passes over the same prompts.

    Attributes:
        num_prompts: Number of prompts compared.
        concurrency_a: Batch concurrency of the first pass.
        concurrency_b: Batch concurrency of the second pass.
        exact_match_rate: Fraction of prompts with identical outputs.
        gen_kl_prefix_mean: Mean generation-path KL on the common prefix, over prompts where it is
            defined, or ``None``.
        flip_eligible: Prompts with a gold label, where a flip can be measured.
        flips: Prompts whose correctness differs between the two passes.
        flip_rate: ``flips / flip_eligible``, or ``None`` without eligible prompts.
        flip_rate_lower: Lower Wilson bound of the flip rate.
        flip_rate_upper: Upper Wilson bound of the flip rate.
    """

    num_prompts: int
    concurrency_a: int
    concurrency_b: int
    exact_match_rate: float | None
    gen_kl_prefix_mean: float | None
    flip_eligible: int
    flips: int
    flip_rate: float | None
    flip_rate_lower: float | None
    flip_rate_upper: float | None


def _output_key(result: CompletionResult) -> list[int] | str:
    """Return the value used to compare two outputs for equality.

    Args:
        result: A completion.

    Returns:
        The token ids when available, otherwise the text.
    """
    return result.output_ids if result.output_ids is not None else result.output_text


def compare_passes(
    first: Sequence[CompletionResult],
    second: Sequence[CompletionResult],
    golds: Sequence[tuple[str | None, str | None]],
    *,
    concurrency_a: int,
    concurrency_b: int,
    confidence: float,
) -> ConsistencyReport:
    """Compare two generation passes over the same prompts.

    Args:
        first: Completions of the first pass, in prompt order.
        second: Completions of the second pass, in prompt order.
        golds: ``(gold, answer_type)`` for each prompt, in the same order.
        concurrency_a: Batch concurrency of the first pass.
        concurrency_b: Batch concurrency of the second pass.
        confidence: Confidence level of the flip-rate interval.

    Returns:
        The consistency report.
    """
    exact_flags: list[bool] = []
    kls: list[float] = []
    eligible = 0
    flips = 0
    for a, b, (gold, answer_type) in zip(first, second, golds, strict=True):
        exact_flags.append(_output_key(a) == _output_key(b))
        if a.output_ids is not None and b.output_ids is not None:
            kl = gen_kl_prefix_mean(a.gen_topk, b.gen_topk, a.output_ids, b.output_ids)
            if kl is not None:
                kls.append(kl)
        correct_a = check_answer(a.output_text, gold, answer_type)
        correct_b = check_answer(b.output_text, gold, answer_type)
        flip, _ = flip_status(correct_a, correct_b)
        if flip is not None:
            eligible += 1
            flips += int(flip)

    total = len(exact_flags)
    exact_rate = sum(exact_flags) / total if total else None
    kl_mean = sum(kls) / len(kls) if kls else None
    rate: float | None = None
    lower: float | None = None
    upper: float | None = None
    if eligible:
        rate = flips / eligible
        lower, upper = wilson_interval(flips, eligible, confidence)
    return ConsistencyReport(
        num_prompts=total,
        concurrency_a=concurrency_a,
        concurrency_b=concurrency_b,
        exact_match_rate=exact_rate,
        gen_kl_prefix_mean=kl_mean,
        flip_eligible=eligible,
        flips=flips,
        flip_rate=rate,
        flip_rate_lower=lower,
        flip_rate_upper=upper,
    )


async def run_consistency_check(
    client: InferenceClient,
    requests: Sequence[GenRequest],
    golds: Sequence[tuple[str | None, str | None]],
    *,
    concurrency_a: int,
    concurrency_b: int,
    logprobs_k: int,
    seed: int,
    confidence: float,
) -> ConsistencyReport:
    """Run the generation path at two concurrencies and compare the outputs.

    Args:
        client: Server client.
        requests: Prompts to generate for, in order.
        golds: ``(gold, answer_type)`` per request.
        concurrency_a: Batch concurrency of the first pass.
        concurrency_b: Batch concurrency of the second pass.
        logprobs_k: Top-k logprobs to request.
        seed: Run seed.
        confidence: Confidence level of the flip-rate interval.

    Returns:
        The consistency report.
    """
    first = await generate_outputs(
        client,
        requests,
        concurrency=concurrency_a,
        logprobs_k=logprobs_k,
        seed=seed,
        mode=GenerationMode(allow_fallback=False),
    )
    second = await generate_outputs(
        client,
        requests,
        concurrency=concurrency_b,
        logprobs_k=logprobs_k,
        seed=seed,
        mode=GenerationMode(allow_fallback=False),
    )
    return compare_passes(
        first,
        second,
        golds,
        concurrency_a=concurrency_a,
        concurrency_b=concurrency_b,
        confidence=confidence,
    )
