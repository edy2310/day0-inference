"""Pure performance metrics (§11.3): per-request times, per-level summaries, cost, and SLO checks.

Times are in milliseconds. Throughput uses the measurement window, which runs from the first
measured request start to the last measured request end.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from day0.client import StreamResult
from day0.config import HardwareConfig, SLOConfig
from day0.stats import mean_or_none, percentile

_NS_PER_MS = 1e6
_SECONDS_PER_HOUR = 3600.0
_TOKENS_PER_MILLION = 1e6


@dataclass(frozen=True)
class RequestTimes:
    """Latency measurements of one successful request.

    Attributes:
        ttft_ms: Time from send to first content chunk.
        tpot_ms: Mean time per output token after the first, or ``None`` for one-token outputs.
        e2e_ms: Time from send to end of stream.
        itl_ms: Gap per output token, one entry per token.
        output_tokens: Output tokens reported by the final usage object.
    """

    ttft_ms: float
    tpot_ms: float | None
    e2e_ms: float
    itl_ms: list[float] = field(default_factory=list)
    output_tokens: int = 0


def request_times(result: StreamResult) -> RequestTimes | None:
    """Compute the latency metrics of one request.

    Args:
        result: The streamed result.

    Returns:
        The metrics, or ``None`` for failed requests or requests without a first token.
    """
    if not result.ok or result.first_token_ns is None:
        return None
    ttft_ms = (result.first_token_ns - result.start_ns) / _NS_PER_MS
    e2e_ms = (result.end_ns - result.start_ns) / _NS_PER_MS
    tpot_ms: float | None = None
    if result.output_tokens > 1:
        tpot_ms = (result.end_ns - result.first_token_ns) / _NS_PER_MS / (result.output_tokens - 1)
    itl: list[float] = []
    for index in range(1, len(result.chunk_ns)):
        count = result.chunk_tokens[index]
        if count <= 0:
            continue
        gap = (result.chunk_ns[index] - result.chunk_ns[index - 1]) / _NS_PER_MS / count
        itl.extend([gap] * count)
    return RequestTimes(
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        e2e_ms=e2e_ms,
        itl_ms=itl,
        output_tokens=result.output_tokens,
    )


def cost_per_million_output_tokens(
    gpu_hour_usd: float, num_gpus: int, output_tok_per_s: float
) -> float | None:
    """Dollar cost per million output tokens at a given throughput.

    Example: $3.00 per GPU-hour, one GPU, 2,500 tokens/s gives about $0.3333 per million tokens.

    Args:
        gpu_hour_usd: Price of one GPU-hour.
        num_gpus: GPUs per server.
        output_tok_per_s: Output throughput.

    Returns:
        The cost, or ``None`` when throughput is zero.
    """
    if output_tok_per_s <= 0:
        return None
    return gpu_hour_usd * num_gpus / (output_tok_per_s * _SECONDS_PER_HOUR) * _TOKENS_PER_MILLION


def _stats(values: Sequence[float]) -> dict[str, float | None]:
    """Compute the percentiles and mean that the report shows.

    Args:
        values: Sample values.

    Returns:
        A mapping with ``p50``, ``p95``, ``p99``, and ``mean``.
    """
    return {
        "p50": percentile(values, 50.0),
        "p95": percentile(values, 95.0),
        "p99": percentile(values, 99.0),
        "mean": mean_or_none(values),
    }


def level_summary(
    *,
    concurrency: int,
    results: Sequence[StreamResult],
    window_s: float,
    slo: SLOConfig,
    hardware: HardwareConfig,
    max_error_rate: float,
) -> dict[str, object]:
    """Summarize one concurrency level.

    A level is invalid when its error rate exceeds ``max_error_rate`` or when no request succeeded.
    Goodput counts a request whose TTFT meets the SLO and whose TPOT meets the SLO. A request with
    an undefined TPOT (one output token) counts as meeting the TPOT target.

    Args:
        concurrency: Closed-loop concurrency of the level.
        results: Measured results, failed ones included.
        window_s: Measurement window in seconds.
        slo: Service-level objective.
        hardware: Price and GPU count.
        max_error_rate: Maximum tolerated error rate.

    Returns:
        A JSON-compatible dictionary with every field of the level summary.
    """
    ok = [result for result in results if result.ok]
    failed = len(results) - len(ok)
    times = [t for t in (request_times(result) for result in ok) if t is not None]
    total_output = sum(result.output_tokens for result in ok)
    tok_per_s = total_output / window_s if window_s > 0 else 0.0
    requests_per_s = len(ok) / window_s if window_s > 0 else 0.0
    ttft = [t.ttft_ms for t in times]
    tpot = [t.tpot_ms for t in times if t.tpot_ms is not None]
    itl = [gap for t in times for gap in t.itl_ms]
    e2e = [t.e2e_ms for t in times]
    good = sum(
        1
        for t in times
        if t.ttft_ms <= slo.ttft_p99_ms and (t.tpot_ms is None or t.tpot_ms <= slo.tpot_p99_ms)
    )
    goodput = good / window_s if window_s > 0 else 0.0
    error_rate = failed / len(results) if results else 1.0

    invalid_reason: str | None = None
    if not ok:
        invalid_reason = "no request succeeded"
    elif error_rate > max_error_rate:
        invalid_reason = f"error rate {error_rate:.4f} exceeds max_error_rate {max_error_rate}"

    ttft_stats = _stats(ttft)
    tpot_stats = _stats(tpot)
    p99_ttft = ttft_stats["p99"]
    p99_tpot = tpot_stats["p99"]
    slo_met = (
        invalid_reason is None
        and p99_ttft is not None
        and p99_tpot is not None
        and p99_ttft <= slo.ttft_p99_ms
        and p99_tpot <= slo.tpot_p99_ms
    )
    return {
        "concurrency": concurrency,
        "n_requests": len(results),
        "n_ok": len(ok),
        "n_failed": failed,
        "error_rate": error_rate,
        "window_s": window_s,
        "output_tokens": total_output,
        "output_tok_per_s": tok_per_s,
        "requests_per_s": requests_per_s,
        "goodput_rps": goodput,
        "ttft_ms": ttft_stats,
        "tpot_ms": tpot_stats,
        "itl_ms": _stats(itl),
        "e2e_ms": _stats(e2e),
        "cost_per_1m_output_tokens_usd": cost_per_million_output_tokens(
            hardware.gpu_hour_usd, hardware.num_gpus, tok_per_s
        ),
        "slo_met": slo_met,
        "invalid": invalid_reason is not None,
        "invalid_reason": invalid_reason,
        "chunk_token_counts_exact": all(result.chunk_counts_exact for result in ok),
    }
