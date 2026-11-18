"""Server metrics: snapshots, counter deltas, speculative-decoding acceptance, and cache gauges.

Metric names differ between vLLM versions, so every lookup is tolerant: a missing metric yields
``None`` and never fails a run.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from day0.client import InferenceClient

SPEC_DRAFTS = "vllm:spec_decode_num_drafts_total"
SPEC_DRAFT_TOKENS = "vllm:spec_decode_num_draft_tokens_total"
SPEC_ACCEPTED = "vllm:spec_decode_num_accepted_tokens_total"
_COUNTER_SUFFIX = "_total"
_CACHE_MARKERS = ("kv_cache_usage", "cache_usage")


@dataclass(frozen=True)
class SpecDecodeSummary:
    """Speculative-decoding acceptance over a measurement window.

    Attributes:
        drafts: Number of draft steps.
        draft_tokens: Number of draft tokens proposed.
        accepted_tokens: Number of draft tokens accepted.
        acceptance_rate: ``accepted_tokens / draft_tokens``.
        mean_acceptance_length: ``1 + accepted_tokens / drafts``.
    """

    drafts: float
    draft_tokens: float
    accepted_tokens: float
    acceptance_rate: float
    mean_acceptance_length: float


def metric_name(key: str) -> str:
    """Return the metric name of a sample key, without its labels.

    Args:
        key: Sample key, for example ``name{a="b"}`` or ``name``.

    Returns:
        The part before the first ``{``.
    """
    return key.split("{", 1)[0]


async def snapshot(client: InferenceClient) -> dict[str, float] | None:
    """Read and parse ``/metrics``.

    Args:
        client: Server client.

    Returns:
        The parsed samples, or ``None`` when the endpoint does not exist.
    """
    return await client.metrics()


def delta(before: Mapping[str, float], after: Mapping[str, float]) -> dict[str, float]:
    """Difference of counters between two snapshots.

    Only counters (names ending in ``_total``) are subtracted. A counter absent from ``before`` is
    treated as zero there.

    Args:
        before: Snapshot taken before the window.
        after: Snapshot taken after the window.

    Returns:
        Counter deltas keyed like the snapshots.
    """
    return {
        key: value - before.get(key, 0.0)
        for key, value in after.items()
        if metric_name(key).endswith(_COUNTER_SUFFIX)
    }


def sum_by_name(samples: Mapping[str, float], name: str) -> float | None:
    """Sum every sample of one metric name across its label sets.

    Args:
        samples: Parsed samples.
        name: Metric name without labels.

    Returns:
        The sum, or ``None`` when the metric is absent.
    """
    values = [value for key, value in samples.items() if metric_name(key) == name]
    return sum(values) if values else None


def spec_decode_summary(counter_delta: Mapping[str, float]) -> SpecDecodeSummary | None:
    """Acceptance rate and mean acceptance length from counter deltas.

    Args:
        counter_delta: Counter deltas over the window (see :func:`delta`).

    Returns:
        The summary, or ``None`` when a counter is absent or the draft counts are zero.
    """
    drafts = sum_by_name(counter_delta, SPEC_DRAFTS)
    draft_tokens = sum_by_name(counter_delta, SPEC_DRAFT_TOKENS)
    accepted = sum_by_name(counter_delta, SPEC_ACCEPTED)
    if drafts is None or draft_tokens is None or accepted is None:
        return None
    if drafts <= 0 or draft_tokens <= 0:
        return None
    return SpecDecodeSummary(
        drafts=drafts,
        draft_tokens=draft_tokens,
        accepted_tokens=accepted,
        acceptance_rate=accepted / draft_tokens,
        mean_acceptance_length=1.0 + accepted / drafts,
    )


def cache_usage(samples: Mapping[str, float]) -> dict[str, float]:
    """Cache-utilization gauges, if the server exposes any.

    Args:
        samples: Parsed samples from one snapshot.

    Returns:
        The matching samples. Empty when none exist.
    """
    return {
        key: value
        for key, value in samples.items()
        if any(marker in metric_name(key) for marker in _CACHE_MARKERS)
    }
