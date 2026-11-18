"""Tests for the performance metrics, using synthetic streamed results."""

from __future__ import annotations

import pytest

from day0.client import StreamResult
from day0.config import HardwareConfig, SLOConfig
from day0.perf.metrics import cost_per_million_output_tokens, level_summary, request_times

MS = 1_000_000  # nanoseconds per millisecond


def _result(
    *,
    start_ms: float,
    first_ms: float,
    chunk_ms: list[float],
    chunk_tokens: list[int],
    end_ms: float,
    output_tokens: int,
    ok: bool = True,
) -> StreamResult:
    """Build a synthetic streamed result with times given in milliseconds from the start.

    Args:
        start_ms: Send time.
        first_ms: Time of the first content chunk.
        chunk_ms: Parse time of each content chunk.
        chunk_tokens: Tokens carried by each content chunk.
        end_ms: End of stream.
        output_tokens: Output tokens reported by usage.
        ok: Whether the request succeeded.

    Returns:
        The result, with times in nanoseconds.
    """
    base = 1_000 * MS
    return StreamResult(
        ok=ok,
        error=None if ok else "boom",
        start_ns=base + int(start_ms * MS),
        first_token_ns=base + int(first_ms * MS) if ok else None,
        chunk_ns=[base + int(t * MS) for t in chunk_ms],
        chunk_tokens=chunk_tokens,
        output_tokens=output_tokens,
        end_ns=base + int(end_ms * MS),
        chunk_counts_exact=True,
    )


def test_ttft_tpot_and_e2e_for_single_token_chunks() -> None:
    """With one token per chunk, TTFT, TPOT, and E2E follow directly from the timestamps."""
    result = _result(
        start_ms=0,
        first_ms=20,
        chunk_ms=[20, 22, 24, 26, 28],
        chunk_tokens=[1, 1, 1, 1, 1],
        end_ms=28,
        output_tokens=5,
    )
    times = request_times(result)
    assert times is not None
    assert times.ttft_ms == pytest.approx(20.0)
    assert times.e2e_ms == pytest.approx(28.0)
    assert times.tpot_ms == pytest.approx((28 - 20) / 4)
    assert times.itl_ms == pytest.approx([2.0, 2.0, 2.0, 2.0])


def test_itl_repeats_gap_per_token_for_multi_token_chunks() -> None:
    """A chunk with three tokens contributes three equal ITL entries of gap / 3."""
    result = _result(
        start_ms=0, first_ms=10, chunk_ms=[10, 16], chunk_tokens=[1, 3], end_ms=16, output_tokens=4
    )
    times = request_times(result)
    assert times is not None
    assert times.itl_ms == pytest.approx([2.0, 2.0, 2.0])


def test_tpot_is_undefined_for_one_token_outputs() -> None:
    """One output token has no TPOT, so the value is ``None`` and is excluded from percentiles."""
    result = _result(
        start_ms=0, first_ms=5, chunk_ms=[5], chunk_tokens=[1], end_ms=5, output_tokens=1
    )
    times = request_times(result)
    assert times is not None
    assert times.tpot_ms is None


def test_failed_requests_have_no_times() -> None:
    """Failed requests produce no metrics."""
    result = _result(
        start_ms=0, first_ms=0, chunk_ms=[], chunk_tokens=[], end_ms=1, output_tokens=0, ok=False
    )
    assert request_times(result) is None


def test_cost_example_from_specification() -> None:
    """$3.00 per GPU-hour, one GPU, and 2,500 tokens/s cost about $0.3333 per million tokens."""
    assert cost_per_million_output_tokens(3.0, 1, 2500.0) == pytest.approx(0.3333, abs=1e-4)
    assert cost_per_million_output_tokens(3.0, 1, 0.0) is None


def test_level_summary_counts_goodput_and_slo() -> None:
    """Goodput counts requests that meet both SLO bounds, and the SLO is judged on p99.

    Args: none; the values are chosen by hand.
    """
    fast = _result(
        start_ms=0,
        first_ms=10,
        chunk_ms=[10, 12, 14],
        chunk_tokens=[1, 1, 1],
        end_ms=14,
        output_tokens=3,
    )
    slow_ttft = _result(
        start_ms=0,
        first_ms=200,
        chunk_ms=[200, 202, 204],
        chunk_tokens=[1, 1, 1],
        end_ms=204,
        output_tokens=3,
    )
    summary = level_summary(
        concurrency=2,
        results=[fast, slow_ttft],
        window_s=1.0,
        slo=SLOConfig(ttft_p99_ms=100, tpot_p99_ms=10),
        hardware=HardwareConfig(num_gpus=1, gpu_hour_usd=3.0),
        max_error_rate=0.01,
    )
    assert summary["n_ok"] == 2
    assert summary["goodput_rps"] == pytest.approx(1.0)
    assert summary["slo_met"] is False
    assert summary["output_tok_per_s"] == pytest.approx(6.0)
    assert summary["invalid"] is False


def test_level_is_invalid_when_errors_exceed_budget() -> None:
    """A level with more errors than ``max_error_rate`` is marked invalid and cannot meet the SLO.

    Args: none; the values are chosen by hand.
    """
    ok = _result(
        start_ms=0, first_ms=5, chunk_ms=[5, 6], chunk_tokens=[1, 1], end_ms=6, output_tokens=2
    )
    bad = _result(
        start_ms=0, first_ms=0, chunk_ms=[], chunk_tokens=[], end_ms=1, output_tokens=0, ok=False
    )
    summary = level_summary(
        concurrency=1,
        results=[ok, bad],
        window_s=1.0,
        slo=SLOConfig(ttft_p99_ms=100, tpot_p99_ms=10),
        hardware=HardwareConfig(num_gpus=1, gpu_hour_usd=3.0),
        max_error_rate=0.01,
    )
    assert summary["invalid"] is True
    assert summary["slo_met"] is False
    assert summary["error_rate"] == pytest.approx(0.5)


def test_percentiles_are_linear_interpolations() -> None:
    """Percentiles across requests use linear interpolation, as numpy does."""
    results = [
        _result(
            start_ms=0,
            first_ms=float(t),
            chunk_ms=[float(t), float(t) + 1],
            chunk_tokens=[1, 1],
            end_ms=float(t) + 1,
            output_tokens=2,
        )
        for t in (10, 20, 30, 40)
    ]
    summary = level_summary(
        concurrency=1,
        results=results,
        window_s=1.0,
        slo=SLOConfig(ttft_p99_ms=1000, tpot_p99_ms=1000),
        hardware=HardwareConfig(num_gpus=1, gpu_hour_usd=3.0),
        max_error_rate=0.0,
    )
    ttft = summary["ttft_ms"]
    assert isinstance(ttft, dict)
    assert ttft["p50"] == pytest.approx(25.0)
    assert ttft["mean"] == pytest.approx(25.0)
