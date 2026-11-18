"""Tests for server-metric deltas and speculative-decoding summaries."""

from __future__ import annotations

import pytest

from day0.parsing import parse_prometheus
from day0.scrape import cache_usage, delta, spec_decode_summary, sum_by_name

DRAFTS = 'vllm:spec_decode_num_drafts_total{model_name="m"}'
DRAFT_TOKENS = 'vllm:spec_decode_num_draft_tokens_total{model_name="m"}'
ACCEPTED = 'vllm:spec_decode_num_accepted_tokens_total{model_name="m"}'


def test_delta_subtracts_counters_only() -> None:
    """Only ``_total`` counters are differenced, and new labels count from zero."""
    before = {DRAFTS: 10.0, 'vllm:num_requests_running{model_name="m"}': 3.0}
    after = {DRAFTS: 25.0, DRAFT_TOKENS: 40.0, 'vllm:num_requests_running{model_name="m"}': 1.0}
    change = delta(before, after)
    assert change == {DRAFTS: 15.0, DRAFT_TOKENS: 40.0}


def test_spec_decode_summary_computes_rate_and_length() -> None:
    """Acceptance rate is accepted / draft tokens, and mean length is 1 + accepted / drafts."""
    counters = {DRAFTS: 100.0, DRAFT_TOKENS: 400.0, ACCEPTED: 240.0}
    summary = spec_decode_summary(counters)
    assert summary is not None
    assert summary.acceptance_rate == pytest.approx(0.6)
    assert summary.mean_acceptance_length == pytest.approx(3.4)


def test_spec_decode_summary_is_none_when_absent_or_zero() -> None:
    """Missing counters or zero draft counts give ``None``, so the card can say so."""
    assert spec_decode_summary({}) is None
    assert spec_decode_summary({DRAFTS: 0.0, DRAFT_TOKENS: 0.0, ACCEPTED: 0.0}) is None
    assert spec_decode_summary({DRAFTS: 5.0, DRAFT_TOKENS: 20.0}) is None


def test_sum_by_name_adds_label_variants() -> None:
    """Two label sets of the same counter are summed."""
    samples = {f'{DRAFTS[:-1]},engine="0"}}': 2.0, DRAFTS: 3.0, "other_metric": 9.0}
    assert sum_by_name(samples, "vllm:spec_decode_num_drafts_total") == pytest.approx(5.0)
    assert sum_by_name(samples, "absent") is None


def test_cache_usage_finds_gauges_by_name_and_is_empty_otherwise() -> None:
    """Gauges with ``cache_usage`` in their name are returned, and other metrics are not."""
    samples = parse_prometheus(
        'vllm:gpu_cache_usage_perc{model_name="m"} 0.25\n'
        "vllm:kv_cache_usage_perc 0.5\n"
        "vllm:num_requests_running 3\n"
    )
    found = cache_usage(samples)
    assert sorted(found.values()) == [0.25, 0.5]
    assert cache_usage({"vllm:num_requests_running": 3.0}) == {}
